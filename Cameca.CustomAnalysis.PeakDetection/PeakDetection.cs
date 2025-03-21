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

namespace Cameca.CustomAnalysis.PeakDetection;

[DefaultView(PeakDetectionViewModel.UniqueId, typeof(PeakDetectionViewModel))]
internal partial class PeakDetection : BasicCustomAnalysisBase<PeakDetectionProperties>
{
    public const string UniqueId = "Cameca.CustomAnalysis.PeakDetection.PeakDetection";

    private readonly IPyExecutor pyExecutor;
    private readonly IContainerProvider containerProvider;
    private readonly ILogger<PeakDetection> logger;
    private float[]? histogramCounts;

    public static INodeDisplayInfo DisplayInfo { get; } = new NodeDisplayInfo("Peak Detection");

    public ObservableCollection<IRenderData> ChartDataSource { get; } = new();
    public ObservableCollection<Range> Ranges { get; } = new();

    [ObservableProperty]
    private bool requiresPropertyUpdate = false;

    public PeakDetection(
        IStandardAnalysisFilterNodeBaseServices services,
        ResourceFactory resourceFactory,
        IPyExecutor pyExecutor,
        IContainerProvider containerProvider,
        ILogger<PeakDetection> logger)
        : base(services, resourceFactory)
    {
        this.pyExecutor = pyExecutor;
        this.containerProvider = containerProvider;
        this.logger = logger;
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
        Ranges.Clear();
        // Get from cached value for performance if already set
        // The histogram data should only need to be recalculated on data invalidation, and for that we alwasy call the full Update method
        histogramCounts ??= await GetHistogramCounts(cancellationToken);
        var ranges = await PredictRanges(cancellationToken, histogramCounts);
        if (histogramCounts is null || ranges is null) { return false; }
        Ranges.AddRange(ranges.OrderBy(x => x.Lower));

        // Update histogram render data with ranges if exists
        if (ChartDataSource.SingleOrDefault() is IHistogramRenderData massHistogram)
        {
            massHistogram.VerticalSlices = Ranges
            .Select(x => new Slice(x.Lower, x.Upper)
            {
                Color = Colors.Red
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
            float x = Lower + (i * BinWidth);
            float y = histogramCounts[i];
            histPos[i] = new Vector2(x, y);
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
            var typedRanges = Ranges.Select(r => new IonTypeInfoRange("X", IonFormula.Unknown, 0d, r.Lower, r.Upper, Colors.Red));
            var existingRanges = rangeMangager.GetIonRanges();
            var inferredRanges = PeakRangeResolver.FromExistingRanges(typedRanges, existingRanges);
            await rangeMangager.SetIonRanges(inferredRanges);
        }
    }

    internal float Lower { get; } = 0f;
    internal float Upper { get; } = 307.2f;
    internal float BinWidth { get; } = 0.01f;
    // User decimal to avoid floating point errors in binning calculation
    internal int BinCount => (int)Math.Ceiling((new decimal(Upper) - new decimal(Lower)) / new decimal(BinWidth));

    internal async Task<Range[]?> PredictRanges(CancellationToken token, float[]? data)
    {
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
        catch (TaskCanceledException)
        {
            return null;
        }

        // Build APSuiteContext
        var context = new APSuiteContextProvider(
            ionData,
            containerProvider,
            Id);

        // Configure Python delgated execution
        var entryFunction = new EntryFunctionDefinition(new[]
            {
                new ParameterDefinition("context", context),
                new ParameterDefinition("histogram", new HistogramDataProvider(data)),
            },
            functionName: "main");
        var executableModule = new LocalModuleExecutable("PeakDetectionModule", entryFunction);

        var ranges = new CaptureManagedResults<Range[]>((PyObject? pyObj) =>
        {
            if (pyObj is null) { return null; }
            dynamic resArray = pyObj;
            int length = resArray.shape[0].As<int>();
            var res = new Range[length];
            for (int i = 0; i < length; i++)
            {
                float lower = resArray[i][0].As<float>();
                float upper = resArray[i][1].As<float>();
                res[i] = new Range(lower * BinWidth, upper * BinWidth);
            }
            return res;
        });
        var middleware = new IPyExecutorMiddleware[]
        {
            ranges,
        };

        // Run
        try
        {
            await pyExecutor.Execute(executableModule, middleware, token);
            DataStateIsValid = true;
            return ranges.HasResult ? ranges.Value : null;
        }
        catch (TaskCanceledException)
        {
            // Cancellation should not be considered a logged error
        }
        catch (PythonException e)
        {
            logger.LogError(e, "Error running Python");
        }
        DataStateIsValid = false;
        return null;
    }

    internal async Task<float[]?> GetHistogramCounts(CancellationToken token)
    {
        if (await Services.IonDataProvider.GetOwnerIonData(Id, cancellationToken: token) is not { } ionData)
        {
            return null;
        }
        return CreateHistogramCounts(ionData, BinWidth, Lower, BinCount);
    }

    private static float[] CreateHistogramCounts(IIonData ionData, float binWidth, float lower, int binCount)
    {
        var histogram = new float[binCount];
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
}