using Cameca.CustomAnalysis.PythonCore;
using Python.Runtime;
using System;

namespace Cameca.CustomAnalysis.PeakDetection;

internal class HistogramDataProvider : IPyObjectProvider
{
    private readonly ReadOnlyMemory<double> data;

    public HistogramDataProvider(ReadOnlyMemory<double> data)
    {
        this.data = data;
    }

    public PyObject GetPyObject(PyModule scope) => data.ToPython();
}
