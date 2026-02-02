using Cameca.CustomAnalysis.Interface;
using System.ComponentModel.DataAnnotations;

namespace Cameca.CustomAnalysis.PeakDetection;

public class GlobalPeakDetectionProperties
{
    [Display(Description = "Object confidence threshold for detection")]
    public double Confidence { get; set; } =  0.5d;
    
    [Display(Name = "Intersection Over Union", Description = "Intersection over union (IoU) threshold for NMS: higher means get more")]
    public double IntersectionOverUnion { get; set; } = 0.01d;

    [Display(Name = "Max Detections", Description = "Maximum number of detections per image")]
    public int MaxDetections { get; set; } = 2000;

    [Display(Name = "Iterations", Description = "Number of peak detection iterations. Each subsequent iteration removes detected ranges before re-running.")]
    [Range(1, int.MaxValue, ErrorMessage = "Minimum of 1 iteration is required")]
    public int Iterations { get; set; } = 1;
}
