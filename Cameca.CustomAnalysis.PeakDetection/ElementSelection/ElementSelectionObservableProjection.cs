using Cameca.CustomAnalysis.Utilities;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Linq;

namespace Cameca.CustomAnalysis.PeakDetection.ElementSelection;

public class ElementSelectionObservableProjection : ObservableCollection<Element>
{
    private readonly ObservableCollection<ElementSelectionModel> models;
    private bool changing = false;

    public ElementSelectionObservableProjection(ObservableCollection<ElementSelectionModel> models)
    {
        this.models = models;

        foreach (var item in this.models)
        {
            Add(item.Element);
        }

        this.models.CollectionChanged += Models_CollectionChanged;
        this.CollectionChanged += Projection_CollectionChanged;
    }


    private void Models_CollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (!changing)
        {
            changing = true;
            if (e.Action == NotifyCollectionChangedAction.Reset)
            {
                Clear();
                foreach (var item in models)
                {
                    Add(item.Element);
                }
            }
            else
            {
                if (e.NewItems is not null)
                {
                    foreach (ElementSelectionModel item in e.NewItems)
                    {
                        Add(item.Element);
                    }
                }

                if (e.OldItems is not null)
                {
                    foreach (ElementSelectionModel item in e.OldItems)
                    {
                        Remove(item.Element);
                    }
                }
            }
            changing = false;
        }
    }

    private void Projection_CollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (!changing)
        {
            changing = true;
            if (e.Action == NotifyCollectionChangedAction.Reset)
            {
                    changing = true;
                    models.Clear();
                    foreach (Element element in this)
                    {
                        models.Add(new ElementSelectionModel
                        {
                            Element = element,
                        });
                    }
                    changing = false;
            }
            else
            {
                if (e.NewItems is not null)
                {
                    foreach (Element element in e.NewItems)
                    {
                        models.Add(new ElementSelectionModel
                        {
                            Element = element,
                        });
                    }
                }
                if (e.OldItems is not null)
                {
                    foreach (Element element in e.OldItems)
                    {
                        foreach (var model in models.Where(x => x.Element == element).ToList())
                        {
                            models.Remove(model);
                        }
                    }
                }
            }
            changing = false;
        }
    }
}
