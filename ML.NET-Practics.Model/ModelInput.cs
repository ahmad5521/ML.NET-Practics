using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ML.NET_Practics.Model
{

    public class ModelInput
    {
        [ColumnName("id"), LoadColumn(0)]
        public float Id { get; set; }


        [ColumnName("perc_premium_paid_by_cash_credit"), LoadColumn(1)]
        public float Perc_premium_paid_by_cash_credit { get; set; }


        [ColumnName("age_in_days"), LoadColumn(2)]
        public float Age_in_days { get; set; }


        [ColumnName("Income"), LoadColumn(3)]
        public float Income { get; set; }


        [ColumnName("Count_3-6_months_late"), LoadColumn(4)]
        public float Count_3_6_months_late { get; set; }


        [ColumnName("Count_6-12_months_late"), LoadColumn(5)]
        public float Count_6_12_months_late { get; set; }


        [ColumnName("Count_more_than_12_months_late"), LoadColumn(6)]
        public float Count_more_than_12_months_late { get; set; }


        [ColumnName("Marital Status"), LoadColumn(7)]
        public string Marital_Status { get; set; }


        [ColumnName("Veh_Owned"), LoadColumn(8)]
        public float Veh_Owned { get; set; }


        [ColumnName("No_of_dep"), LoadColumn(9)]
        public float No_of_dep { get; set; }


        [ColumnName("Accomodation"), LoadColumn(10)]
        public string Accomodation { get; set; }


        [ColumnName("risk_score"), LoadColumn(11)]
        public float Risk_score { get; set; }


        [ColumnName("no_of_premiums_paid"), LoadColumn(12)]
        public float No_of_premiums_paid { get; set; }


        [ColumnName("sourcing_channel"), LoadColumn(13)]
        public string Sourcing_channel { get; set; }


        [ColumnName("residence_area_type"), LoadColumn(14)]
        public string Residence_area_type { get; set; }


        [ColumnName("premium"), LoadColumn(15)]
        public float Premium { get; set; }


        [ColumnName("default"), LoadColumn(16)]
        public bool Default { get; set; }


    }
}
