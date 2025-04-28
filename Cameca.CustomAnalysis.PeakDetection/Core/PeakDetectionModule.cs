using Cameca.CustomAnalysis.Interface;
using Cameca.CustomAnalysis.PythonCore;
using Cameca.CustomAnalysis.Utilities;
using Prism.Ioc;
using Prism.Modularity;

namespace Cameca.CustomAnalysis.PeakDetection;

/// <summary>
/// Public <see cref="IModule"/> implementation is the entry point for AP Suite to discover and configure the custom analysis
/// </summary>
public class PeakDetectionModule : IModule
{
    public void RegisterTypes(IContainerRegistry containerRegistry)
    {
        containerRegistry.AddCustomAnalysisUtilities(options => options.UseStandardBaseClasses = true);
        containerRegistry.RegisterPythonDistribution();

        containerRegistry.Register<object, PeakDetection>(PeakDetection.UniqueId);
        containerRegistry.RegisterInstance(PeakDetection.DisplayInfo, PeakDetection.UniqueId);
        containerRegistry.Register<IAnalysisMenuFactory, PeakDetectionMenuFactory>(nameof(PeakDetectionMenuFactory));
        containerRegistry.Register<object, PeakDetectionViewModel>(PeakDetectionViewModel.UniqueId);
    }

    public void OnInitialized(IContainerProvider containerProvider)
    {
        var extensionRegistry = containerProvider.Resolve<IExtensionRegistry>();

        extensionRegistry.RegisterAnalysisView<PeakDetectionView, PeakDetectionViewModel>(AnalysisViewLocation.Default);

        containerProvider.InitializePythonDistribution("Peak Detection - Python Configuration");
        extensionRegistry.RegisterOptions<GlobalPeakDetectionProperties>("Peak Detection");
    }
}
