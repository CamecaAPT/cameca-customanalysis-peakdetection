using Cameca.CustomAnalysis.PeakDetection.ElementSelection;
using Cameca.CustomAnalysis.Utilities;
using CommunityToolkit.Mvvm.ComponentModel;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Numerics;

namespace Cameca.CustomAnalysis.PeakDetection;

public partial class PeakDetectionProperties : ObservableValidator 
{
    [ObservableProperty]
    [field: Display(Description = "Object confidence threshold for detection")]
    private double confidence = 0.5d;

    [ObservableProperty]
    [field: Display(Name = "Intersection Over Union", Description = "Intersection over union (IoU) threshold for NMS: higher means get more")]
    private double intersectionOverUnion = 0.01d;

    [ObservableProperty]
    [field: Display(Name = "Max Detections", Description = "Maximum number of detections per image")]
    private int maxDetections = 2000;

    [ObservableProperty]
    [field: Display(Name = "Iterations", Description = "Number of peak detection iterations. Each subsequent iteration removes detected ranges before re-running.")]
    [field: Range(1, int.MaxValue, ErrorMessage = "Minimum of 1 iteration is required")]
    private int iterations = 0;

    [ObservableProperty]
    [field: Display(Name = "Use Peak Maxima", Description = "Use peak maxima for ion type assignment matching, else use left edge of range")]
        private bool usePeakMaxima = true;

    [ObservableProperty]
    [field: Display(AutoGenerateField = false)]
    private List<ElementSelectionModel> elementSelectionModels = new();

    [ObservableProperty]
    [field: Display(AutoGenerateField = false)]
    private Vector2 viewportLower;

    [ObservableProperty]
    [field: Display(AutoGenerateField = false)]
    private Vector2 viewportUpper;

}