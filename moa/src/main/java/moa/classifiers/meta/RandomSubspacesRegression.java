package moa.classifiers.meta;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.FilteredSparseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import java.util.TreeSet;

public class RandomSubspacesRegression extends AbstractClassifier implements Regressor {

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 'e',
            "Size of the ensemble", 10, 1, 10000);

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'b',
            "Base learner type.", Classifier.class, "trees.FIMTDD");

    // RANDOM SUBSPACES (% of features per learner) 1 = all features to all learners
    public FloatOption subspacePercentageOption = new FloatOption("subspacePercentage", 's',
            "Subspace percentage. 1 = all features are associated with each learner.", 1.0,
            0.0, 1.0);

    public FlagOption useBaggingOption = new FlagOption("useBagging", 'B', "Flag to determine whether bagging should be used.");

    public FloatOption lambdaOption = new FloatOption("lambda", 'l', "Lambda parameter for bagging.", 1, 1, 10);

    RSRegressionLearner ensemble[];

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double output = 0.0;
        for(int i = 0; i < ensemble.length; i++){
            output += ensemble[i].getVotesForInstance(inst);
        }
        output /= ensemble.length;
        return new double[]{output};
    }

    @Override
    public void resetLearningImpl() {
        AbstractClassifier base = (AbstractClassifier) getPreparedClassOption(baseLearnerOption);
        this.ensemble = new RSRegressionLearner[ensembleSizeOption.getValue()];
        for(int i = 0; i < ensembleSizeOption.getValue(); i++){
            this.ensemble[i] = new RSRegressionLearner(base.copy(), subspacePercentageOption.getValue());
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        for(RSRegressionLearner l : ensemble){
            int k = 1;
            if(useBaggingOption.isSet()){
                double lambda = lambdaOption.getValue();
                k = MiscUtils.poisson(lambda, this.classifierRandom);
            }
            if (k > 0) {
                for (int ixTimes = 0; ixTimes < k; ixTimes++) {
                    l.trainOnInstance(inst);
                }
            }
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {}

    @Override
    public boolean isRandomizable() {
        return true;
    }

    class RSRegressionLearner{

        private Classifier learner   = null;
        private int featureIndices[] = null;
        private double pctFeatures;

        public RSRegressionLearner(Classifier learner, double pctFeatures) {
            this.learner = learner;
            this.learner.prepareForUse();
            this.learner.resetLearning();
            this.pctFeatures = pctFeatures;
        }

        public double getVotesForInstance(Instance instnc){
            if(featureIndices == null) featureIndices = getFeatureIndices(pctFeatures,
                    instnc.numAttributes() - 1,
                    instnc.classIndex());
            Instance filteredInstnc = filterInstance(instnc);
            return learner.getVotesForInstance(filteredInstnc)[0];
        }

        public void trainOnInstance(Instance instnc){
            Instance filteredInstnc = filterInstance(instnc);
            this.learner.trainOnInstance(filteredInstnc);
        }

        public Instance filterInstance(Instance instnc) {
            Instance filtered;
            if(featureIndices != null && featureIndices.length > 0 &&
                    featureIndices.length < instnc.numAttributes() - 1){
                int numAttributes = instnc.numAttributes();

                // copies all values including the class
                int indices[] = new int[featureIndices.length + 1];
                double values[] = new double[featureIndices.length + 1];
                for (int i = 0; i < featureIndices.length; i++) {
                    indices[i] = featureIndices[i];
                    values[i] = instnc.value(featureIndices[i]);
                }
                //adds the class index and value
                indices[indices.length - 1] = instnc.classIndex();
                values[indices.length - 1] = instnc.classValue();

                filtered = new FilteredSparseInstance(1.0, values, indices, numAttributes);
                filtered.setDataset(instnc.dataset());
            }else{
                filtered = instnc;
            }
            return filtered;
        }

        private int[] getFeatureIndices(double pctFeatures, int numFeatures, int classIndex) {
            // Selects the features that will be used randomly with equal chance
            // The code also ignores the class index
            int numSelectedFeatures = (int) Math.ceil(pctFeatures * numFeatures);
            TreeSet<Integer> selected = new TreeSet<>();
            while(selected.size() < numSelectedFeatures) {
                int position = classifierRandom.nextInt(numFeatures);
                if(position != classIndex){
                    selected.add(position);
                }
            }

            // builds the final array
            int arr[] = new int[numSelectedFeatures];
            int index = 0;
            for(Integer i : selected){
                arr[index] = i;
                index++;
            }
            return arr;
        }

    }
}
