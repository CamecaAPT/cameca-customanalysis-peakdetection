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
using Cameca.CustomAnalysis.PeakDetection.ModelValidation;
using Prism.Services.Dialogs;

namespace Cameca.CustomAnalysis.PeakDetection;

[DefaultView(PeakDetectionViewModel.UniqueId, typeof(PeakDetectionViewModel))]
internal partial class PeakDetection : BasicCustomAnalysisBase<PeakDetectionProperties>
{
    public const string UniqueId = "Cameca.CustomAnalysis.PeakDetection.PeakDetection";

    private readonly IContainerProvider containerProvider;
    private readonly ILogger<PeakDetection> logger;
    private readonly PythonService pythonService;
    private readonly ModelValidator modelValidator;
    private readonly IDialogService dialogService;
    private double[]? histogramCounts;

    public static INodeDisplayInfo DisplayInfo { get; } = new NodeDisplayInfo("Peak Detection");

    public ObservableCollection<IRenderData> ChartDataSource { get; } = new();
    public ObservableCollection<IonTypeInfoRangeRowInfo> Ranges { get; } = new();

    [ObservableProperty]
    private bool requiresPropertyUpdate = false;

    public PeakDetection(
        IStandardAnalysisFilterNodeBaseServices services,
        ResourceFactory resourceFactory,
        IContainerProvider containerProvider,
        ILogger<PeakDetection> logger,
        PythonService pythonService,
        ModelValidator modelValidator,
        IDialogService dialogService)
        : base(services, resourceFactory)
    {
        this.containerProvider = containerProvider;
        this.logger = logger;
        this.pythonService = pythonService;
        this.modelValidator = modelValidator;
        this.dialogService = dialogService;
    }

    protected override void OnCreated(NodeCreatedEventArgs eventArgs)
    {
        base.OnCreated(eventArgs);
        if (eventArgs.Trigger == EventTrigger.Create && Resources.Options.GetOptions<GlobalPeakDetectionProperties>() is { } propDefaults)
        {
            Properties.Implementation = propDefaults.DefaultMethod;
            Properties.ModelPath = propDefaults.DefaultModelPath;
            Properties.ScalarPath = propDefaults.DefaultScalarPath;
            Properties.Confidence = propDefaults.Confidence;
            Properties.IntersectionOverUnion = propDefaults.IntersectionOverUnion;
            Properties.MaxDetections = propDefaults.MaxDetections;
        }
    }

    partial void OnRequiresPropertyUpdateChanged(bool value) => OnPropertyChanged(nameof(UpdateCommandCanExecute));

    protected override void OnPropertiesChanged(PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(PeakDetectionProperties.ViewportLower) || e.PropertyName == nameof(PeakDetectionProperties.ViewportUpper))
        {
        }
        else
        {
            base.OnPropertiesChanged(e);
            RequiresPropertyUpdate = true;
        }
    }

    public override bool UpdateCommandCanExecute => base.UpdateCommandCanExecute || RequiresPropertyUpdate;


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
            var name = results.Res[i];
            var confidence = results.Confidence[i];
            string? name2 = results.Res2[i];
            name2 = name2 != "NaN" ? name2 : null;
            float? confidence2 = results.Confidence2[i];
            confidence2 = confidence2 > 0 ? confidence2 : null;
            var infoRange = Resources.CreateIonTypeInfoRange(name, rng.Lower, rng.Upper);
            var rowInfo = new IonTypeInfoRangeRowInfo(
                infoRange, confidence,
                name2, confidence2);
            unsorted.Add(rowInfo);
        }
        Ranges.Clear();
        foreach (var item in unsorted.OrderBy(x => x.IonTypeInfoRange.Min))
        {
            Ranges.Add(item);
        }

        // Update histogram render data with ranges if exists
        if (ChartDataSource.SingleOrDefault() is IHistogramRenderData massHistogram)
        {
            massHistogram.VerticalSlices = Ranges
                .Select(x => x.IonTypeInfoRange)
                .Select(x => new Slice((float)x.Min, (float)x.Max)
                {
                    Color = x.Color,
                })
                .ToList();
        }
        RequiresPropertyUpdate = false;
        return true;
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
        return await PropetiesDependantUpdate(cancellationToken);
    }

    // Full update - recomputes histogram values. Necessary if data state is invalidated
    protected override async Task<bool> Update(CancellationToken cancellationToken)
    {
        if (DataStateIsValid)
        {
            if (RequiresPropertyUpdate)
            {
                return await PropetiesDependantUpdate(cancellationToken);
            }
        }
        else
        {
            return await FullUpdate(cancellationToken);
        }
        return false;
    }

    [RelayCommand]
    public async Task ApplyPeaks(CancellationToken token)
    {
        if (Resources.RangeManager is { } rangeMangager && Resources.GetMassSpectrum() is not null)
        {
            var discreteRanges = OverlapResolver.RemoveOverlaps(Ranges.Select(x => x.IonTypeInfoRange));
            if (!await rangeMangager.SetIonRanges(discreteRanges))
            {
                logger.LogWarning("Could not apply ranges");
            }
        }
    }

    internal double Lower { get; } = 0d;
    internal double Upper { get; } = 307.2d;
    internal double BinWidth { get; } = 0.01d;
    // User decimal to avoid floating point errors in binning calculation
    internal int BinCount => (int)Math.Ceiling((new decimal(Upper) - new decimal(Lower)) / new decimal(BinWidth));

    internal async Task<PyResults?> PredictRanges(CancellationToken token, double[]? data)
    {
        // Get other data
        if (data is null)
        {
            return null;
        }

        // Validate model files to ensure trust during the Python code deserialization of unsafe pickle files
        if (!ValidateModelFiles())
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
            // Configure Python delgated execution
            var py_results = await pythonService
                .MapPythonFunction("PeakDetectionModule", "main")
                .SetParameters(context, (ReadOnlyMemory<double>)data)
                .Call(token);
            var results = MapToClrObjects(py_results);
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

    private bool ValidateModelFiles()
    {
        var reqsValidation = new List<string>();
        if (Properties.Implementation == PeakDetectionImplementation.NeuralNetwork)
        {
            if (Properties.ModelPath is not null)
            {
                reqsValidation.Add(Properties.ModelPath);
            }
        }
        else if (Properties.Implementation == PeakDetectionImplementation.RandomForest)
        {
            if (Properties.ModelPath is not null)
            {
                reqsValidation.Add(Properties.ModelPath);
            }
            if (Properties.ScalarPath is not null)
            {
                reqsValidation.Add(Properties.ScalarPath);
            }
        }

        foreach (var path in reqsValidation)
        {
            var validationResult = modelValidator.ValidateModelFile(path);
            if (!validationResult.IsTrusted)
            {
                // Prompt that model is untrusted: abort, continue once, or add as trusted and continue
                var result = dialogService.ShowUntrustedModelDialog(path, validationResult.Hash);
                if (result is null || result is { AllowContinue: false })
                {
                    logger.LogError("Could not run peak detection: Model file is not trusted: {ModelPath}", path);
                    if (DataState is not null)
                    {
                        DataState.IsErrorState = true;
                    }
                    return false;
                }

                // Optionally add as trusted file
                if (result.AddTrusted && validationResult.Hash is not null)
                {
                    modelValidator.AddTrustedHash(validationResult.Hash);
                }
            }
        }
        return true;
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
    internal record IonTypeInfoRangeRowInfo(IonTypeInfoRange IonTypeInfoRange, float Confidence, string? Name2, float? Confidence2);
}
