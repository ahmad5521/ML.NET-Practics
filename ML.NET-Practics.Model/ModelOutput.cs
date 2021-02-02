using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ML.NET_Practics.Model
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }

    }
}
