using Cameca.CustomAnalysis.Interface;
using CommunityToolkit.Mvvm.ComponentModel;
using System.Windows.Media;

namespace Cameca.CustomAnalysis.PeakDetection;

public partial class Slice : ObservableObject, IChart2DSlice
{
    public float Min { get; }

    public float Max { get; }

    [ObservableProperty]
    private Color color;

    [ObservableProperty]
    private bool isSelected;

    public Slice(float min, float max)
    {
        Min = min;
        Max = max;
    }
}