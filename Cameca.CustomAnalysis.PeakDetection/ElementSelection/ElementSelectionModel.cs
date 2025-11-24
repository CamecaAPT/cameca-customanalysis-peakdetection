using Cameca.CustomAnalysis.Utilities;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Cameca.CustomAnalysis.PeakDetection.ElementSelection;

public sealed partial class ElementSelectionModel : ObservableObject
{
    public Element Element { get; set; }

    [ObservableProperty]
    private bool includeComplex = false;
}
