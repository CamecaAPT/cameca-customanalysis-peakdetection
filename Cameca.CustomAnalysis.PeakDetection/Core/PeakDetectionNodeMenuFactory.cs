using Cameca.CustomAnalysis.Interface;
using Cameca.CustomAnalysis.Utilities;
using Prism.Events;

namespace Cameca.CustomAnalysis.PeakDetection;

internal class PeakDetectionMenuFactory : AnalysisMenuFactoryBase
{
    public PeakDetectionMenuFactory(IEventAggregator eventAggregator)
        : base(eventAggregator)
    {
    }

    protected override INodeDisplayInfo DisplayInfo => PeakDetection.DisplayInfo;
    protected override string NodeUniqueId => PeakDetection.UniqueId;
    public override AnalysisMenuLocation Location { get; } = AnalysisMenuLocation.Analysis;
}