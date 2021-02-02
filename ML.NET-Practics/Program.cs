using System;
using ML.NET_Practics.Model;
using System.Collections.Generic;

namespace ML.NET_Practics
{

    class Program
    {
        static void Main(string[] args)
        {
            ModelBuilder.CreateModel();


            List<ModelInput> list = new List<ModelInput>() {
               new ModelInput(){
                    Id = 1F,
                    Perc_premium_paid_by_cash_credit = 0.317F,
                    Age_in_days = 11330F,
                    Income = 90050F,
                    Count_3_6_months_late = 0F,
                    Count_6_12_months_late = 0F,
                    Count_more_than_12_months_late = 0F,
                    Marital_Status = @"0",
                    Veh_Owned = 3F,
                    No_of_dep = 3F,
                    Accomodation = @"1",
                    Risk_score = 98.81F,
                    No_of_premiums_paid = 8F,
                    Sourcing_channel = @"A",
                    Residence_area_type = @"Rural",
                    Premium = 5400,
                    Default = true
               },
               new ModelInput(){
                    Id = 8F,
                    Perc_premium_paid_by_cash_credit = 0.994F,
                    Age_in_days = 14248F,
                    Income = 84090F,
                    Count_3_6_months_late = 0F,
                    Count_6_12_months_late = 0F,
                    Count_more_than_12_months_late = 0F,
                    Marital_Status = @"0",
                    Veh_Owned = 3F,
                    No_of_dep = 3F,
                    Accomodation = @"1",
                    Risk_score = 98.99F,
                    No_of_premiums_paid = 4F,
                    Sourcing_channel = @"A",
                    Residence_area_type = @"Urban",
                    Premium = 3300,
                    Default = true
               },
               new ModelInput(){
                    Id = 5F,
                    Perc_premium_paid_by_cash_credit = 0.888F,
                    Age_in_days = 19360F,
                    Income = 103050F,
                    Count_3_6_months_late = 7F,
                    Count_6_12_months_late = 3F,
                    Count_more_than_12_months_late = 4F,
                    Marital_Status = @"0",
                    Veh_Owned = 2F,
                    No_of_dep = 1F,
                    Accomodation = @"1",
                    Risk_score = 98.8F,
                    No_of_premiums_paid = 15F,
                    Sourcing_channel = @"A",
                    Residence_area_type = @"Urban",
                    Premium = 1000,
                    Default = false
               },
               new ModelInput(){
                    Id = 183F,
                    Perc_premium_paid_by_cash_credit = 0.958F,
                    Age_in_days = 16798F,
                    Income = 364080F,
                    Count_3_6_months_late = 1F,
                    Count_6_12_months_late = 3F,
                    Count_more_than_12_months_late = 1F,
                    Marital_Status = @"1",
                    Veh_Owned = 0F,
                    No_of_dep = 1F,
                    Accomodation = @"2",
                    Risk_score = 99.09F,
                    No_of_premiums_paid = 7F,
                    Sourcing_channel = @"d",
                    Residence_area_type = @"Rural",
                    Premium = 13800f,
                    Default = false
               }
            };

            Console.WriteLine("=============== Using  model ===============");
            Console.WriteLine("Using model to make multi prediction -- \nComparing actual Default value with predicted Default value from sample data...");
            foreach (var item in list)
            {
                var predictionResult = ConsumeModel.Predict(item);

                Console.WriteLine($"Id: {item.Id}");
                Console.WriteLine($"Actual value: {item.Default}");
                Console.WriteLine($"Predicted Value(Label): {predictionResult.PredictedLabel}");
                Console.WriteLine($"Predicted Value(Score): {predictionResult.Score}");
            }

            Console.WriteLine("=============================================");
        }
    }
}
