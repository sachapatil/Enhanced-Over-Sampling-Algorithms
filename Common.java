/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package algo;

import java.io.File;
import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

/**
 *
 * @author user
 */
public class Common {

    public static double percentage = 0.4;
    public static boolean filterappply = false;
    public static int option = 2;  //option 1 : LVH , 2 : OVA ,: Binnary:  0
    public static int index = 0;
    public static int mmmcnousalgo = 0; //  
    public static Filter[] filter = new Filter[]{new SMOTE(), new MEMMOT(), new MMMmOT(), new CMEOT(), new NFNMOT(), new MMCBUOST(), new UCPMOST()};
    public static ArrayList<InstanceIndex> classindex = new ArrayList<InstanceIndex>();

    public static RandomForest rf = new RandomForest();
    public static NaiveBayes nb = new NaiveBayes();
    public static MultilayerPerceptron mp = new MultilayerPerceptron();
    public static AdaBoostM1 am = new AdaBoostM1();

  public static   String[] algooption = new String[]{"-I 100 -num-slots 1 -K 0 -S 1",
        "", "-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a",
        "-P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump"};

    static {
        try {

            rf.setOptions(new String[]{"-I", "100", "-num-slots", "1", "-K", "0", "-S", "1"});
             //rf.setOptions(new String[]{"-P", "100", "-I", "100", "-num-slots", "1", "-K", "0", "-M", "1.0", "-V", "0.001", "-S", "1"});
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        try {
            nb.setOptions(new String[]{});
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        try {
            mp.setOptions(new String[]{"-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "a"});
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        try {
            am.setOptions(new String[]{"-P", "100", "-S", "1", "-I", "10", "-W", "weka.classifiers.trees.DecisionStump"});
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
    public static Classifier[] c = new Classifier[]{rf, nb, mp, am};

    public static class InstanceIndex {

        public Instances in;
        public int index;

        public InstanceIndex(Instances in, int index) {
            this.in = in;
            this.index = index;
        }

    }

    public Common() {
    }

}
