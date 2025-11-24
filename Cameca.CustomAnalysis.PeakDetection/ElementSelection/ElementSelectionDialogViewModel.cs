using Cameca.CustomAnalysis.Utilities;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Prism.Services.Dialogs;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cameca.CustomAnalysis.PeakDetection.ElementSelection;

internal partial class ElementSelectionDialogViewModel : ObservableObject, IDialogAware
{
    public ObservableCollection<Element> Elements { get; }
    public ObservableCollection<ElementSelectionModel> Models { get; } = new();

    public ElementSelectionDialogViewModel()
    {
        Elements = new ElementSelectionObservableProjection(Models);
    }

    public string Title { get; } = "Select Elements";

    public event Action<IDialogResult>? RequestClose;

    public bool CanCloseDialog() => true;

    [RelayCommand]
    public void Ok() => RequestClose?.Invoke(new DialogResult(ButtonResult.OK, new DialogParameters
    {
        { "Models", Models.ToList() }
    }));

    [RelayCommand]
    public void Close() => RequestClose?.Invoke(new DialogResult(ButtonResult.Cancel));

    public void OnDialogClosed() { }

    public void OnDialogOpened(IDialogParameters parameters)
    {
        if (parameters.TryGetValue("Models", out List<ElementSelectionModel> initModels))
        {
            Models.Clear();
            foreach (var model in initModels)
            {
                Models.Add(model);
            }
        }
    }
}
