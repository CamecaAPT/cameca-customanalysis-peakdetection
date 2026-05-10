using Cameca.CustomAnalysis.Interface;
using Cameca.CustomAnalysis.PeakDetection.ElementSelection;
using Cameca.CustomAnalysis.PythonCore;
using Cameca.CustomAnalysis.Utilities;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json.Linq;
using Prism.Ioc;
using Prism.Services.Dialogs;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;

namespace Cameca.CustomAnalysis.PeakDetection;

[DefaultView(PeakDetectionViewModel.UniqueId, typeof(PeakDetectionViewModel))]
[NodeType(NodeType.Analysis)]
internal partial class PeakDetection : BasicCustomAnalysisBase<PeakDetectionProperties>
{
    public RecommendElementsProperties RecommendationProperties { get; set; } = new();

    [ObservableProperty]
    private List<Element> recommendedElements = new();


    private PythonRpcSession<HostCallbacks, IPythonApi> session;
    private bool disposed;
    private readonly MemMapStore memMapStore = new();
    //private readonly ILoggerFactory loggerFactory;

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

    private readonly ILogger<PeakDetection> logger;
    private readonly PythonSessionFactory pythonSessionFactory;
    private double[]? histogramCounts;

    public static INodeDisplayInfo DisplayInfo { get; } = new NodeDisplayInfo("Peak Detection");

    public ObservableCollection<IRenderData> ChartDataSource { get; } = new();
    public ObservableCollection<IonTypeInfoRangeRowInfo> Ranges { get; } = new();

    public PeakDetection(
        IStandardAnalysisFilterNodeBaseServices services,
        ResourceFactory resourceFactory,
        ILogger<PeakDetection> logger,
        //ILoggerFactory loggerFactory,
        PythonSessionFactory pythonSessionFactory)
        : base(services, resourceFactory)
    {
        this.logger = logger;
        this.pythonSessionFactory = pythonSessionFactory;
        //this.loggerFactory = loggerFactory;

        EnsurePythonSessionStarted();
    }

    [MemberNotNull(nameof(session))]
    private void EnsurePythonSessionStarted()
    {
        if (disposed)
        {
            throw new ObjectDisposedException(GetType().FullName);
        }
        session ??= pythonSessionFactory.Start<HostCallbacks, IPythonApi>(
            Path.Join("PythonModules", "PeakDetectionModule.py"),
            new HostCallbacks(
                logger,
                //loggerFactory.CreateLogger("PeakDetectionModule.py"),
                Resources,
                memMapStore));
    }
    protected override void Dispose(bool disposing)
    {
        session.Dispose();
        base.Dispose(disposing);
    }

    protected override void OnCreated(NodeCreatedEventArgs eventArgs)
    {
        base.OnCreated(eventArgs);
        if (eventArgs.Trigger == EventTrigger.Create && Resources.Options.GetOptions<GlobalPeakDetectionProperties>() is { } propDefaults)
        {
            Properties.Confidence = propDefaults.Confidence;
            Properties.IntersectionOverUnion = propDefaults.IntersectionOverUnion;
            Properties.MaxDetections = propDefaults.MaxDetections;
            Properties.Iterations = propDefaults.Iterations;
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
            int iteration = results.Iterations[i];
            confidence2 = confidence2 > 0 ? confidence2 : null;
            var infoRange = Resources.CreateIonTypeInfoRange(name, rng.Lower, rng.Upper);
            var infoRange2 = name2 is not null ? Resources.CreateIonTypeInfoRange(name2, rng.Lower, rng.Upper) : null;
            var rowInfo = new IonTypeInfoRangeRowInfo(
                infoRange, key, confidence,
                infoRange2, key2, confidence2, iteration, include: iteration == 1);
            rowInfo.PropertyChanged += RowInfo_PropertyChanged;
            unsorted.Add(rowInfo);
        }

        // Copy old ranges to merge user selections where possible
        var oldInfo = Ranges.ToArray();
        Ranges.Clear();
        foreach (var item in unsorted.OrderBy(x => x.IonTypeInfoRange.Min))
        {
            // Find if the range overlaps with any of the old existing ranges
            var prevItem = oldInfo.FirstOrDefault(oldRow => RangeUtils.RangesOverlap(oldRow.IonTypeInfoRange, item.IonTypeInfoRange));
            if (prevItem is not null)
            {
                // Get the 1st/2nd selection
                var prevInfoRange = prevItem.Use2 ? prevItem.IonTypeInfoRange2 : prevItem.IonTypeInfoRange;
                // If the selected item type matches the 2nd of the new type, selec the Use2 option to keep that type 
                if (item.IonTypeInfoRange2 is not null && item.IonTypeInfoRange2.Formula.Equals(prevInfoRange?.Formula))
                {
                    item.Use2 = true;
                }

                // Regardless of type matching, try to maintain the inclusion selection
                item.Include = prevItem.Include;
            }

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
                .Where(x => x.Include)
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
        if (sender is IonTypeInfoRangeRowInfo { } row
            && (e.PropertyName == nameof(IonTypeInfoRangeRowInfo.Use2) || e.PropertyName == nameof(IonTypeInfoRangeRowInfo.Include)))
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
                components[name] += int.TryParse(count, out var intCount) ? intCount : 1;
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

    private async Task LoadDataModel(CancellationToken cancellationToken)
    {
        await Resources.Progress.ShowDialog("Loading data", async (p, t) =>
        {
            var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, t);
            await session.Remote.PreloadTrainingData(p, cts.Token);
        });
    }


    [RelayCommand]
    public async Task ApplyPeaks(CancellationToken token)
    {
        if (Resources.RangeManager is { } rangeMangager && Resources.GetMassSpectrum() is not null)
        {
            var ranges = Ranges.Where(x => x.Include).Select(x => x.Use2 ? x.IonTypeInfoRange2! : x.IonTypeInfoRange);
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
        // Run
        try
        {
            await LoadDataModel(token);
            var results = await Resources.Progress.ShowDialog("Recommending Elements", async () =>
            {
                return await session.Remote.RecommendElements(
                    (Ranges.Where(x => x.Include).Select(x => new double[] { x.IonTypeInfoRange.Min, x.IonTypeInfoRange.Max }).ToArray()),
                    Ranges.Where(x => x.Include).Select(x => x.Use2 ? x.Key2! : x.Key).ToArray(),
                    RecommendationProperties.Threshold,
                    RecommendationProperties.NumElements);
            });

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
        catch (Exception e)
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
        // Run
        try
        {
            await LoadDataModel(token);
            string[] elementList = Properties.ElementSelectionModels.Select(x => x.Element.ToString()).ToArray();
            string[] elementsOtMolecules = Properties.ElementSelectionModels.Where(x => x.IncludeComplex).Select(x => x.Element.ToString()).ToArray();
            double confidence = Properties.Confidence;
            double intersectionOverUnion = Properties.IntersectionOverUnion;
            int maxDetections = Properties.MaxDetections;
            int iterations = Properties.Iterations;

            var res = await session.Remote.PredictRanges(
                elementList,
                elementsOtMolecules,
                confidence,
                intersectionOverUnion,
                maxDetections,
                iterations);

            var results = new PyResults(
                res.PeakPred.Select(x => new Range(x[0], x[1])).ToArray(),
                res.Elem1,
                res.Conf1,
                res.Elem2,
                res.Conf2,
                res.PeakIter);
            DataStateIsValid = true;
            return results;
        }
        catch (OperationCanceledException)
        {
            // Cancellation should not be considered a logged error
        }
        catch (Exception e)
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
                }
                ;
            }
        }
        return histogram;
    }

    internal record Range(float Lower, float Upper);
    internal record PyResults(Range[] Ranges, string[] Res, float[] Confidence, string[] Res2, float[] Confidence2, int[] Iterations);
    internal partial class IonTypeInfoRangeRowInfo : ObservableObject
    {
        [ObservableProperty]
        private bool use2;

        [ObservableProperty]
        private bool include = true;

        public IonTypeInfoRangeRowInfo(IonTypeInfoRange IonTypeInfoRange, string Key, float Confidence, IonTypeInfoRange? IonTypeInfoRange2, string? Key2, float? Confidence2, int Iteration, bool use2 = false, bool include = true)
        {
            this.IonTypeInfoRange = IonTypeInfoRange;
            this.IonTypeInfoRange2 = IonTypeInfoRange2;
            this.Confidence = Confidence;
            this.Confidence2 = Confidence2;
            this.Key = Key;
            this.Key2 = Key2;
            this.Iteration = Iteration;
            Use2 = use2;
            Include = include; ;
        }

        public IonTypeInfoRange IonTypeInfoRange { get; }
        public string Key { get; }
        public float Confidence { get; }
        public IonTypeInfoRange? IonTypeInfoRange2 { get; }
        public string? Key2 { get; }
        public float? Confidence2 { get; }
        public int Iteration { get; }
    }
}

public class RecommendElementsProperties
{
    [Display(Name = "Confidence Threshold", Description = "Assignments with confidence below this threshold are candidates to be considered for alternative elements")]
    public double Threshold { get; set; } = 0.01;

    [Display(Name = "Max Recommendations/Peak", Description = "Maximum number of alternative elements that can be proposed for any single peak")]
    public int NumElements { get; set; } = 3;
}