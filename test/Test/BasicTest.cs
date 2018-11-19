
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NPatternRecognizer.Algorithm.ANN;
using NPatternRecognizer.Algorithm.Boost;
using NPatternRecognizer.Algorithm.KNN;
using NPatternRecognizer.Algorithm.SVM;
using NPatternRecognizer.Common;
using NPatternRecognizer.Interface;

namespace NPatternRecognizer.Test
{
    /// <summary>
    /// NUnit Test Attribute Execution Order:
    /// 
    /// 1. TestFixtureSetUp 
    /// 2. SetUp 
    /// 3. Test 
    /// 4. TearDown 
    /// 5. Repeat steps 2, 3, and 4 for each test that's being run in this fixture. 
    /// 6. TestFixtureTearDown
    /// 
    /// 
    /// </summary>
    [TestClass]
    public class BasicTest
    {
        #region SVM

        [TestMethod]
        public void Binary_SVM_SMO_Test()
        {
            Logger.Info("Binary_SVM_SMO_Test");

            var problem = ProblemFactory.CreateClassificationProblem(ClassificationProblemType.ChessBoard);

            var tSet = problem.TrainingSet;
            var vSet = problem.ValidationSet;

            var classifier = new Binary_SVM_SMO(problem)
            {
                TrainSet = tSet,
                Kernel = new GaussianRBFKernel(0.001)
            };


            classifier.Train();

            Logger.Info("Doing cross-validation.");

            var hit = (from e in vSet.Examples let iResult = classifier.Predict(e.X) where e.Label.Id == iResult select e).Count();

            var correctRatio = 1.0 * hit / vSet.Count;

            Assert.IsTrue(correctRatio > 0.950, string.Format("SVM-SMO (2-class) Correct Ratio, expected: greater than 0.970, actual: {0}.", correctRatio));
        }

        #endregion


        #region KNN

        [TestMethod]
        public void KNN_Test()
        {
            Logger.Info("KNN_Test");

            var problem = ProblemFactory.CreateClassificationProblem(ClassificationProblemType.ChessBoard);

            Logger.Info("Loading training data.");

            var tSet = problem.TrainingSet;
            var vSet = problem.ValidationSet;

            var classifier = new KNN
            {
                KNN_K = 7,
                TrainSet = tSet
            };
            classifier.Train();

            Logger.Info("Doing cross-validation.");

            var hit = (from e in vSet.Examples let iResult = classifier.Predict(e.X) where e.Label.Id == iResult select e).Count();

            var correctRatio = 1.0 * hit / vSet.Count;

            Logger.Info("CorrectRatio: {0}", correctRatio);

            Assert.IsTrue(correctRatio > 0.900, string.Format("KNN (2-class) Correct Ratio, expected: greater than 0.900, actual: {0}.", correctRatio));
        }

        #endregion


        #region ANN_BP

        [TestMethod]
        public void ANN_BP_Test()
        {
            Logger.Info("ANN_BP_Test");

            var problem = ProblemFactory.CreateClassificationProblem(ClassificationProblemType.ChessBoard);

            Logger.Info("Loading training data.");

            var tSet = problem.TrainingSet;
            var vSet = problem.ValidationSet;

            var classifier = new ANN_BP(problem.Dimension)
            {
                TrainSet = tSet,
                MaximumIteration = int.MaxValue,
                ANN_Eta = 0.5,
                ANN_Epsilon = 1e-3,
                LogInterval = 1000
            };

            classifier.Train();

            Logger.Info("Doing cross-validation.");

            var hit = (from e in vSet.Examples let iResult = classifier.Predict(e.X) where e.Label.Id == iResult select e).Count();

            var correctRatio = 1.0 * hit / vSet.Count;

            Logger.Info("CorrectRatio: {0}", correctRatio);

            Assert.IsTrue(correctRatio > 0.900, string.Format("ANN_BP (2-class) Correct Ratio, expected: greater than 0.930, actual: {0}.", correctRatio));
        }

        #endregion


        #region AdaBoost

        [TestMethod]
        public void AdaBoost_Test()
        {
            Logger.Info("AdaBoost_Test");

            var problem = ProblemFactory.CreateClassificationProblem(ClassificationProblemType.ChessBoard);

            var tSet = problem.TrainingSet;
            var vSet = problem.ValidationSet;

            var classifier = new AdaBoost(tSet, 1000, problem.Dimension);
            classifier.Train();

            var hit = (from e in vSet.Examples let iResult = classifier.Predict(e.X) where e.Label.Id == iResult select e).Count();

            var correctRatio = 1.0 * hit / vSet.Count;

            Logger.Info("CorrectRatio: {0}", correctRatio);

            Assert.IsTrue(correctRatio > 0.900, string.Format("AdaBoost (2-class) Correct Ratio, expected: greater than 0.900, actual: {0}.", correctRatio));
        }

        #endregion

    }
}
