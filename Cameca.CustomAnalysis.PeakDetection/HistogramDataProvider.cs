using Cameca.CustomAnalysis.PythonCore;
using Python.Runtime;
using System;

namespace Cameca.CustomAnalysis.PeakDetection;

internal class HistogramDataProvider : IPyObjectProvider
{
    private readonly ReadOnlyMemory<float> data;

    public HistogramDataProvider(ReadOnlyMemory<float> data)
    {
        this.data = data;
    }

    public PyObject GetPyObject(PyModule scope) => data.ToPython();
}
