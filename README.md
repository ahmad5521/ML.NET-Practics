# ML.NET-Practics
The Idea Came when I see the Library and feel cerucity to try it!!

![Image of start](https://i.ytimg.com/vi/x-XPfTA8Glk/maxresdefault.jpg)


In my Senarie [which already done using R](https://ahmasirier.medium.com/executive-summary-problem-statement-premium-paid-by-the-customer-is-the-major-revenue-source-for-a21f3be88f0) i want to applay Binary classification to predict the apility of premium default fro new potintial customer
## loading data From Text File


          IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                          path: path,
                                          hasHeader: true,
                                          separatorChar: ',',
                                          allowQuoting: true,
                                          allowSparse: false);
                                          
## Build training pipeline

            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[]
                {
                    new InputOutputColumnPair("Marital Status", "Marital Status"),
                    new InputOutputColumnPair("Accomodation", "Accomodation"),
                    new InputOutputColumnPair("sourcing_channel", "sourcing_channel"),
                    new InputOutputColumnPair("residence_area_type", "residence_area_type")
                })
                .Append(mlContext.Transforms.Categorical.OneHotHashEncoding(new[]
                {
                    new InputOutputColumnPair("Count_3-6_months_late", "Count_3-6_months_late"),
                    new InputOutputColumnPair("Count_6-12_months_late", "Count_6-12_months_late"),
                    new InputOutputColumnPair("Count_more_than_12_months_late", "Count_more_than_12_months_late"),
                    new InputOutputColumnPair("Veh_Owned", "Veh_Owned"),
                    new InputOutputColumnPair("No_of_dep", "No_of_dep")
                }))
                .Append(mlContext.Transforms.Concatenate("Features", new[] {
                    "Count_3-6_months_late",
                    "Count_6-12_months_late",
                    "Count_more_than_12_months_late",
                    "Income",
                    "risk_score",
                    "Marital Status",
                    "Accomodation",
                    "sourcing_channel",
                    "residence_area_type",
                    "premium",
                    "Veh_Owned",
                    "No_of_dep",
                    "perc_premium_paid_by_cash_credit",
                    "age_in_days",
                    "no_of_premiums_paid"
                }));


            var trainer = mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: @"default", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);
           
## Training  model


            ITransformer model = trainingPipeline.Fit(trainingDataView);


## Evaluate Non Calibrated to get model's accuracy metrics

            IDataView predictions = model.Transform(trainingDataView);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(predictions, "default", "Score");
            Console.WriteLine($"*   Metrics for Binary Classfication model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*   Accuracy:        {metrics.Accuracy:0.##}");
            Console.WriteLine($"*   AreaUnderPrecisionRecallCurve:        {metrics.AreaUnderPrecisionRecallCurve:0.##}");
            Console.WriteLine($"*   AreaUnderRocCurve:        {metrics.AreaUnderRocCurve:0.##}");
            Console.WriteLine($"*   F1Score:        {metrics.F1Score:0.##}");
            Console.WriteLine($"*   NegativePrecision:        {metrics.NegativePrecision:0.##}");
            Console.WriteLine($"*   NegativeRecall:        {metrics.NegativeRecall:0.##}");
            Console.WriteLine($"*   PositivePrecision:        {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"*   PositiveRecall:        {metrics.PositiveRecall:0.##}");



## Using  model
Using model to make multi prediction, Comparing actual Default value with predicted Default value from sample data...

            foreach (var item in list)
            {
                var predictionResult = ConsumeModel.Predict(item);

                Console.WriteLine($"Id: {item.Id}");
                Console.WriteLine($"Actual value: {item.Default}");
                Console.WriteLine($"Predicted Value(Label): {predictionResult.PredictedLabel}");
                Console.WriteLine($"Predicted Value(Score): {predictionResult.Score}");
            }
            
            
            
            
## Result

![Image of Result](https://github.com/ahmad5521/ML.NET-Practics/blob/master/ML.NET-Practics/Data/Capture.JPG?raw=true)

