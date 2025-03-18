using Cameca.CustomAnalysis.Utilities;

namespace Cameca.CustomAnalysis.PeakDetection;

internal class PeakDetectionViewModel : AnalysisViewModelBase<PeakDetection>
{
    public const string UniqueId = "Cameca.CustomAnalysis.PeakDetection.PeakDetectionViewModel";

    public PeakDetectionViewModel(IAnalysisViewModelBaseServices services)
        : base(services)
    {
    }
}