using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML;
using ML.NET_Practics.Model;

namespace ML.NET_Practics
{
    public static class ModelBuilder
    {
        private static string TRAIN_DATA_FILE = @"Insurance Premium Default-Dataset.csv";
        private static string MODEL_FILE = ConsumeModel.MLNetModelPath;

        // Create MLContext to be shared across the model creation workflow objects 
        // Set a random seed for repeatable/deterministic results across multiple trainings.
        private static MLContext mlContext = new MLContext(seed: 1);

        public static void CreateModel()
        {

            var path = Path.Combine(Directory.GetParent(Directory.GetParent(Directory.GetParent(Directory.GetCurrentDirectory()).FullName).FullName).FullName, "Data", TRAIN_DATA_FILE);
            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: path,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Build training pipeline
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);

            // Train Model
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            // Evaluate quality of Model
            Evaluate(mlContext, trainingDataView, mlModel);


            // Save model
            SaveModel(mlContext, mlModel, MODEL_FILE, trainingDataView.Schema);
        }

        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
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

            return trainingPipeline;
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine($"=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            return model;
        }

        private static void Evaluate(MLContext mlContext, IDataView trainingDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluate Non Calibrated to get model's accuracy metrics ===============");

            IDataView predictions = model.Transform(trainingDataView);
            var metrics = mlContext.BinaryClassification.
              EvaluateNonCalibrated(predictions, "default", "Score");


            PrintRegressionFoldsAverageMetrics(metrics);
        }

        private static void PrintRegressionFoldsAverageMetrics(BinaryClassificationMetrics metrics)
        {
            //Console.WriteLine($"*************************************************");
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

            //Console.WriteLine($"*************************************************");
        }

        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            //Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            //Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }


    }
}
