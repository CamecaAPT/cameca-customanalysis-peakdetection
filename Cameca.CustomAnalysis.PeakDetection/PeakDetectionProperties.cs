using CommunityToolkit.Mvvm.ComponentModel;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Numerics;

namespace Cameca.CustomAnalysis.PeakDetection;

public partial class PeakDetectionProperties : ObservableObject
{
    [ObservableProperty]
    [field: Display(Description = "Object confidence threshold for detection")]
    private double confidence = 0.2d;

    [ObservableProperty]
    [field: Display(Name = "Intersection Over Union", Description = "Intersection over union (IoU) threshold for NMS: higher means get more")]
    private double intersectionOverUnion = 0.01d;

    [ObservableProperty]
    [field: Display(Name = "Max Detections", Description = "Maximum number of detections per image")]
    private int maxDetections = 2000;

    [ObservableProperty]
    [field: Display(AutoGenerateField = false)]
    private Vector2 viewportLower;

    [ObservableProperty]
    [field: Display(AutoGenerateField = false)]
    private Vector2 viewportUpper;
}
