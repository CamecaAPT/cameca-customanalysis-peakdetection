using Cameca.CustomAnalysis.Interface;
using Cameca.CustomAnalysis.Utilities;
using System.Threading;
using System.Threading.Tasks;

namespace Cameca.CustomAnalysis.PeakDetection;

[DefaultView(PeakDetectionViewModel.UniqueId, typeof(PeakDetectionViewModel))]
internal partial class PeakDetection : BasicCustomAnalysisBase<PeakDetectionProperties>
{
    public const string UniqueId = "Cameca.CustomAnalysis.PeakDetection.PeakDetection";
    
    public static INodeDisplayInfo DisplayInfo { get; } = new NodeDisplayInfo("Peak Detection");

    public PeakDetection(IStandardAnalysisFilterNodeBaseServices services, ResourceFactory resourceFactory)
        : base(services, resourceFactory)
    {
    }

    protected override Task<bool> Update(CancellationToken cancellationToken)
    {
        return base.Update(cancellationToken);
    }
}