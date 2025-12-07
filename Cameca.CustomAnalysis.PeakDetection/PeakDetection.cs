using Cameca.CustomAnalysis.Interface;
using Cameca.CustomAnalysis.PythonCore;
using Cameca.CustomAnalysis.Utilities;
using CommunityToolkit.Mvvm.Input;
using Prism.Ioc;
using Python.Runtime;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;
using Microsoft.Extensions.Logging;
using CommunityToolkit.Mvvm.ComponentModel;
using System.Collections.Generic;
using Prism.Services.Dialogs;
using Cameca.CustomAnalysis.PeakDetection.ElementSelection;
using System.Text.RegularExpressions;
using System.ComponentModel.DataAnnotations;

namespace Cameca.CustomAnalysis.PeakDetection;

[DefaultView(PeakDetectionViewModel.UniqueId, typeof(PeakDetectionViewModel))]
[NodeType(NodeType.Analysis)]
internal partial class PeakDetection : BasicCustomAnalysisBase<PeakDetectionProperties>
{
    private CachedDataModel? dataModel = null;
    public RecommendElementsProperties RecommendationProperties { get; set; } = new();

    [ObservableProperty]
    private List<Element> recommendedElements = new();

    [RelayCommand]
    public void SelectElements()
    {
        Services.DialogService.ShowDialog(
            nameof(ElementSelectionDialogViewModel),
            new DialogParameters
            {
                { "Models", Properties.ElementSelectionModels }
            },
            (results) =>
            {
                if (results.Result == ButtonResult.OK && results.Parameters.TryGetValue("Models", out List<ElementSelectionModel> models))
                {
                    Properties.ElementSelectionModels = models;
                }
            });
    }

    public const string UniqueId = "Cameca.CustomAnalysis.PeakDetection.PeakDetection";

    private readonly IContainerProvider containerProvider;
    private readonly ILogger<PeakDetection> logger;
    private readonly PythonService pythonService;
    private double[]? histogramCounts;

    public static INodeDisplayInfo DisplayInfo { get; } = new NodeDisplayInfo("Peak Detection");

    public ObservableCollection<IRenderData> ChartDataSource { get; } = new();
    public ObservableCollection<IonTypeInfoRangeRowInfo> Ranges { get; } = new();

    public PeakDetection(
        IStandardAnalysisFilterNodeBaseServices services,
        ResourceFactory resourceFactory,
        IContainerProvider containerProvider,
        ILogger<PeakDetection> logger,
        PythonService pythonService)
        : base(services, resourceFactory)
    {
        this.containerProvider = containerProvider;
        this.logger = logger;
        this.pythonService = pythonService;
    }

    protected override void OnCreated(NodeCreatedEventArgs eventArgs)
    {
        base.OnCreated(eventArgs);
        if (eventArgs.Trigger == EventTrigger.Create && Resources.Options.GetOptions<GlobalPeakDetectionProperties>() is { } propDefaults)
        {
            Properties.Confidence = propDefaults.Confidence;
            Properties.IntersectionOverUnion = propDefaults.IntersectionOverUnion;
            Properties.MaxDetections = propDefaults.MaxDetections;
            Properties.ElementSelectionModels = GetInitialElements();
        }
    }

    private List<ElementSelectionModel> GetInitialElements()
    {
        var dict = new Dictionary<Element, bool>();
        if (Resources?.GetMassSpectrum() is { } massSpec && massSpec.GetValidIonData() is { } ionData)
        {
            foreach (var ion in ionData.Ions)
            {
                bool isComplex = ion.Formula.Count > 1 || (ion.Formula.Count == 1 && ion.Formula.First().Value > 1);
                foreach (var elem in ion.Formula)
                {
                    if (Enum.TryParse<Element>(elem.Key, out var elemEnum))
                    {
                        dict[elemEnum] = isComplex;
                    }
                }
            }
        }
        return dict.Select(x => new ElementSelectionModel
        {
            Element = x.Key,
            IncludeComplex = x.Value,
        }).ToList();
    }

    protected override void OnPropertiesChanged(PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(PeakDetectionProperties.ViewportLower) || e.PropertyName == nameof(PeakDetectionProperties.ViewportUpper))
        {
        }
        else
        {
            SetRunPeakDetectionCommandCanExecute(true);
        }
        CanSave = true;
    }

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RunPeakDetectionCommand))]
    private bool peakDetectionDirty = false;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RunPeakDetectionCommand))]
    private bool runPeakDetectionCommandCanExecute = true;

    [RelayCommand(CanExecute = nameof(RunPeakDetectionCommandCanExecute))]
    public async Task RunPeakDetection(CancellationToken cancellationToken)
    {
        var res = !(await PropetiesDependantUpdate(cancellationToken));
        PeakDetectionDirty = false;
        SetRunPeakDetectionCommandCanExecute(res);
    }

    private void SetRunPeakDetectionCommandCanExecute(bool hasChanges)
    {
        // Requires at lease one element selected to run
        RunPeakDetectionCommandCanExecute = hasChanges && Properties.ElementSelectionModels.Any();
        PeakDetectionDirty |= hasChanges;
    }

    private async Task<bool> PropetiesDependantUpdate(CancellationToken cancellationToken)
    {
        var unsorted = new List<IonTypeInfoRangeRowInfo>();
        // Get from cached value for performance if already set
        // The histogram data should only need to be recalculated on data invalidation, and for that we alwasy call the full Update method
        histogramCounts ??= await GetHistogramCounts(cancellationToken);
        var results = await PredictRanges(cancellationToken, histogramCounts);
        if (histogramCounts is null || results is null) { return false; }
        for (int i = 0; i < results.Ranges.Length; i++)
        {
            var rng = results.Ranges[i];
            var key = results.Res[i];
            var name = StandardizeName(key);
            var confidence = results.Confidence[i];
            string? key2 = results.Res2[i];
            string? name2 = key2 != "NaN" ? StandardizeName(key2) : null;
            float? confidence2 = results.Confidence2[i];
            confidence2 = confidence2 > 0 ? confidence2 : null;
            var infoRange = Resources.CreateIonTypeInfoRange(name, rng.Lower, rng.Upper);
            var infoRange2 = name2 is not null ? Resources.CreateIonTypeInfoRange(name2, rng.Lower, rng.Upper) : null;
            var rowInfo = new IonTypeInfoRangeRowInfo(
                infoRange, key, confidence,
                infoRange2, key2, confidence2);
            rowInfo.PropertyChanged += RowInfo_PropertyChanged;
            unsorted.Add(rowInfo);
        }
        Ranges.Clear();
        foreach (var item in unsorted.OrderBy(x => x.IonTypeInfoRange.Min))
        {
            Ranges.Add(item);
        }

        UpdateHistogramSlices();
        return true;
    }

    private void UpdateHistogramSlices()
    {
        // Update histogram render data with ranges if exists
        if (ChartDataSource.SingleOrDefault() is IHistogramRenderData massHistogram)
        {
            massHistogram.VerticalSlices = Ranges
                .Select(x =>
                {
                    var data = x.IonTypeInfoRange;
                    return new Slice((float)data.Min, (float)data.Max)
                    {
                        Color = x.Use2 ? x.IonTypeInfoRange2!.Color : data.Color,
                    };
                })
                .ToList();
        }
    }

    private void RowInfo_PropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (sender is IonTypeInfoRangeRowInfo { } row && e.PropertyName == nameof(IonTypeInfoRangeRowInfo.Use2))
        {
            UpdateHistogramSlices();
        }

    }

    /// <summary>
    /// Python results apparently include some complex formulas with duplicate elements, so normalize to our requirement of one element per ion formula
    /// </summary>
    /// <param name="rawName"></param>
    /// <returns></returns>
    private string StandardizeName(string rawName)
    {
        if (!IonFormulaEx.TryParse(rawName, out var formula))
        {
            var components = new Dictionary<string, int>();
            foreach (Match item in new Regex("(?<name>[A-Z][a-z]?)(?<count>[2-9]|[1-9][0-9]+)?").Matches(rawName))
            {
                string name = item.Groups["name"].Value;
                string count = item.Groups["count"].Value;
                
                if (!components.ContainsKey(name))
                {
                    components[name] = 0;
                }
                components[name] += int.TryParse(count, out var intCount) ? intCount: 1;
            }
            var form = new IonFormula(components.Select((pair) => new IonFormula.Component(pair.Key, pair.Value)));
            return form.ToString();
        }
        return rawName;
    }

    private async Task<bool> FullUpdate(CancellationToken cancellationToken)
    {
        ChartDataSource.Clear();
        Ranges.Clear();
        histogramCounts = await GetHistogramCounts(cancellationToken);
        if (histogramCounts is null) { return false; }
        var histPos = new Vector2[BinCount];
        for (int i = 0; i < BinCount; i++)
        {
            double x = Lower + (i * BinWidth);
            double y = histogramCounts[i];
            histPos[i] = new Vector2((float)x, (float)y);
        }

        var massHistogram = Resources.ChartObjects.CreateHistogram(histPos, color: Colors.Black, thickness: 1f);
        ChartDataSource.Add(massHistogram);
        return true;
        //return await PropetiesDependantUpdate(cancellationToken);
    }

    // Full update - recomputes histogram values. Necessary if data state is invalidated
    protected override async Task<bool> Update(CancellationToken cancellationToken)
    {
        var res = await FullUpdate(cancellationToken);
        SetRunPeakDetectionCommandCanExecute(true);
        return res;
    }

    private async Task<CachedDataModel> LoadDataModel(CancellationToken token)
    {
        return await Resources.Progress.ShowDialog("Loading data", async (p, t) =>
        {
            var cts = CancellationTokenSource.CreateLinkedTokenSource(token, t);
            // Configure Python delgated execution
            var py_results = await pythonService
                .MapPythonFunction("PeakDetectionModule", "load_full_training_data")
                .SetParameters(new object[] { p })
                .Call(cts.Token);
            var results = MapToClrDataModel(py_results);
            return results;
        });
    }

    private CachedDataModel MapToClrDataModel(PyObject? pyObj)
    {
        if (pyObj is null) { throw new InvalidOperationException("Incorrect return format from 'load_full_training_data'"); }
        dynamic resArray = pyObj;

        int[] encodedIons = resArray[0].As<int[]>();
        double[] counts = resArray[1].As<double[]>();
        double[] massToChart = resArray[2].As<double[]>();
        var encoder = new Dictionary<string, int>();
        var py_encoder = resArray[3];
        foreach (var item in py_encoder.items())
        {
            encoder[item[0].As<string>()] = item[1].As<int>();
        }
        var decoder = new Dictionary<int, string>();
        var py_decoder = resArray[4];
        foreach (var item in py_decoder.items())
        {
            decoder[item[0].As<int>()] = item[1].As<string>();
        }

        return new CachedDataModel(encodedIons, counts, massToChart, encoder, decoder);
    }


    [RelayCommand]
    public async Task ApplyPeaks(CancellationToken token)
    {
        if (Resources.RangeManager is { } rangeMangager && Resources.GetMassSpectrum() is not null)
        {
            var ranges = Ranges.Select(x => x.Use2 ? x.IonTypeInfoRange2! : x.IonTypeInfoRange);
            var discreteRanges = OverlapResolver.RemoveOverlaps(ranges);
            if (!await rangeMangager.SetIonRanges(discreteRanges))
            {
                logger.LogWarning("Could not apply ranges");
            }
        }
    }

    [RelayCommand]
    public async Task RecommendElements(CancellationToken token)
    {
        // Ensure data model is cached
        dataModel ??= await LoadDataModel(token);

        // Run
        try
        {
            var results = await Resources.Progress.ShowDialog("Recommending Elements", async (t) =>
            {
                var cte = CancellationTokenSource.CreateLinkedTokenSource(t, token);
                // Configure Python delgated execution
                var py_results = await pythonService
                    .MapPythonFunction("PeakDetectionModule", "recommend_elements")
                    .SetReloadOnCall(true)
                    .SetParameters(
                        ((ReadOnlyMemory<int>)(dataModel.EncodedIons)),
                        ((ReadOnlyMemory<double>)(dataModel.Counts)),
                        ((ReadOnlyMemory<double>)(dataModel.MassToCharge)),
                        dataModel.Encoder,
                        dataModel.Decoder,
                        (ReadOnlyMemory<double>)(Ranges.SelectMany(x => new double[] { x.IonTypeInfoRange.Min, x.IonTypeInfoRange.Max }).ToArray()),
                        Ranges.Select(x => x.Use2 ? x.Key2! : x.Key).ToArray(),
                        RecommendationProperties.Threshold,
                        RecommendationProperties.NumElements,
                        (string msg) => logger.LogDebug(msg))
                    .Call(cte.Token);
                string[] res = py_results!.As<string[]>();
                return res;
            });
            //if (results is null && DataState is not null)
            //{
            //    DataState.IsErrorState = true;
            //}
            //DataStateIsValid = true;
            //return results;

            var newRecElem = new List<Element>();
            foreach (var x in results)
            {
                if (Enum.TryParse<Element>(x, out var e) && !Properties.ElementSelectionModels.Any(x => x.Element == e))
                {
                    newRecElem.Add(e);
                }
            }
            RecommendedElements = newRecElem;
        }
        catch (OperationCanceledException)
        {
            // Cancellation should not be considered a logged error
        }
        catch (PythonException e)
        {
            logger.LogError(e, "Error running Python");
            if (DataState is not null)
            {
                DataState.IsErrorState = true;
            }
        }
    }

    [RelayCommand]
    public void AddRecommend()
    {
        // Copy
        var newSelectionModels = Properties.ElementSelectionModels.ToList();
        bool anyAdded = false;
        foreach (var elem in recommendedElements)
        {
            // Don't duplicate
            if (!newSelectionModels.Any(x => x.Element == elem))
            {
                anyAdded = true;
                newSelectionModels.Add(new ElementSelectionModel
                {
                    Element = elem,
                    IncludeComplex = false,
                });
            }
        }
        if (anyAdded)
        {
            Properties.ElementSelectionModels = newSelectionModels;
        }
    }

    internal double Lower { get; } = 0d;
    internal double Upper { get; } = 307.2d;
    internal double BinWidth { get; } = 0.01d;
    // User decimal to avoid floating point errors in binning calculation
    internal int BinCount => (int)Math.Ceiling((new decimal(Upper) - new decimal(Lower)) / new decimal(BinWidth));

    internal async Task<PyResults?> PredictRanges(CancellationToken token, double[]? data)
    {
        // Ensure data model is cached
        dataModel ??= await LoadDataModel(token);

        // Get other data
        if (data is null)
        {
            return null;
        }

        // Get IonData to provide to APSuiteContext
        IIonData? ionData;
        try
        {
            ionData = await Services.IonDataProvider.GetOwnerIonData(Id, cancellationToken: token);
            if (ionData is null)
            {
                return null;
            }
        }
        catch (OperationCanceledException)
        {
            return null;
        }

        // Build APSuiteContext
        var context = new APSuiteContextProvider(
            ionData,
            containerProvider,
            Id,
            Resources);
        // Run
        try
        {
            //var results = await Resources.Progress.ShowDialog("Running Peak Detection", async (t) =>
            //{
            //    var cte = CancellationTokenSource.CreateLinkedTokenSource(t, token);
            // Configure Python delgated execution
            var py_results = await pythonService
                .MapPythonFunction("PeakDetectionModule", "main")
                .SetParameters(
                    context,
                    (ReadOnlyMemory<double>)data,
                    Properties.ElementSelectionModels.Select(x => x.Element.ToString()).ToArray(),
                    Properties.ElementSelectionModels.Where(x => x.IncludeComplex).Select(x => x.Element.ToString()).ToArray(),
                    Properties.Confidence,
                    Properties.IntersectionOverUnion,
                    Properties.MaxDetections,
                    ((ReadOnlyMemory<int>)(dataModel.EncodedIons)),
                    ((ReadOnlyMemory<double>)(dataModel.Counts)),
                    ((ReadOnlyMemory<double>)(dataModel.MassToCharge)),
                    dataModel.Encoder,
                    dataModel.Decoder,
                    Properties.UsePeakMaxima,
                    (string msg) => logger.LogDebug(msg))
                .Call(token);
                var results = MapToClrObjects(py_results);
            //});
            if (results is null && DataState is not null)
            {
                DataState.IsErrorState = true;
            }
            DataStateIsValid = true;
            return results;
        }
        catch (OperationCanceledException)
        {
            // Cancellation should not be considered a logged error
        }
        catch (PythonException e)
        {
            logger.LogError(e, "Error running Python");
            if (DataState is not null)
            {
                DataState.IsErrorState = true;
            }
        }
        DataStateIsValid = false;
        return null;
    }

    private PyResults? MapToClrObjects(PyObject? pyObj)
    {
        if (pyObj is null) { return null; }
        dynamic resArray = pyObj;

        // predicted peaks
        float[][] pyRangeData = resArray[0].As<float[][]>();
        var peak_pred = pyRangeData
            .Select(rng => new Range((float)(rng[0] * BinWidth), (float)(rng[1] * BinWidth)))
            .ToArray();

        // res
        int length = (int)pyObj[1].Length();
        var res = new string[length];
        for (int i = 0; i < length; i++)
        {
            res[i] = resArray[1][i].ToString();
        }

        // confidence
        float[] confidence = resArray[2].As<float[]>();

        var res2 = new string[length];
        if (resArray[3].IsNone())
        {
            Array.Fill(res2, "");
        }
        else
        {
            for (int i = 0; i < length; i++)
            {
                res2[i] = resArray[3][i].ToString();
            }
        }

        // confidence
        float[] confidence2 = new float[length];
        if (resArray[4].IsNone())
        {
            Array.Fill(confidence2, 0f);
        }
        else
        {
            confidence2 = resArray[4].As<float[]>();
        }

        return new PyResults(peak_pred, res, confidence, res2, confidence2);
    }

    internal async Task<double[]?> GetHistogramCounts(CancellationToken token)
    {
        if (await Services.IonDataProvider.GetOwnerIonData(Id, cancellationToken: token) is not { } ionData)
        {
            return null;
        }
        return CreateHistogramCounts(ionData, BinWidth, Lower, BinCount);
    }

    private static double[] CreateHistogramCounts(IIonData ionData, double binWidth, double lower, int binCount)
    {
        var histogram = new double[binCount];
        foreach (var chunk in ionData.CreateSectionDataEnumerable(IonDataSectionName.Mass))
        {
            var mass = chunk.ReadSectionData<float>(IonDataSectionName.Mass).Span;
            for (int i = 0; i < chunk.Length; i++)
            {
                // Cast to int 
                int binIndex = (int)Math.Floor((mass[i] - lower) / binWidth);
                if (binIndex >= 0 && binIndex < binCount)
                {
                    histogram[binIndex]++;
                };
            }
        }
        return histogram;
    }

    internal record Range(float Lower, float Upper);
    internal record PyResults(Range[] Ranges, string[] Res, float[] Confidence, string[] Res2, float[] Confidence2);
    internal partial class IonTypeInfoRangeRowInfo : ObservableObject
    {
        [ObservableProperty]
        private bool use2;

        public IonTypeInfoRangeRowInfo(IonTypeInfoRange IonTypeInfoRange, string Key, float Confidence, IonTypeInfoRange? IonTypeInfoRange2, string? Key2, float? Confidence2, bool use2 = false)
        {
            this.IonTypeInfoRange = IonTypeInfoRange;
            this.IonTypeInfoRange2 = IonTypeInfoRange2;
            this.Confidence = Confidence;
            this.Confidence2 = Confidence2;
            this.Key = Key;
            this.Key2 = Key2;
            Use2 = use2;
        }

        public IonTypeInfoRange IonTypeInfoRange { get; }
        public string Key { get; }
        public float Confidence { get; }
        public IonTypeInfoRange? IonTypeInfoRange2 { get; }
        public string? Key2 { get; }
        public float? Confidence2 { get; }
    }

    internal class CachedDataModel
    {
        public CachedDataModel(int[] encodedIons, double[] counts, double[] massToChart, Dictionary<string, int> encoder, Dictionary<int, string> decoder)
        {
            EncodedIons = encodedIons;
            Counts = counts;
            MassToCharge = massToChart;
            Encoder = encoder;
            Decoder = decoder;
        }

        public int[] EncodedIons { get; }
        public double[] Counts { get; }
        public double[] MassToCharge { get; }
        public Dictionary<string, int> Encoder { get; }
        public Dictionary<int, string> Decoder { get; }
    }
}

public class RecommendElementsProperties
{
    [Display(Name = "Confidence Threshold", Description = "Assignments with confidence below this threshold are candidates to be considered for alternative elements")]
    public double Threshold { get; set; } = 0.01;

    [Display(Name = "Max Recommendations/Peak", Description = "Maximum number of alternative elements that can be proposed for any single peak")]
    public int NumElements { get; set; } = 3;
}