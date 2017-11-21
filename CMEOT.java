/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package algo;

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * SMOTE.java
 * 
 * Copyright (C) 2008 Ryan Lichtenwalter 
 * Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 */
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.UnsupportedClassTypeException;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

/**
 * <!-- globalinfo-start -->
 * Resamples a dataset by applying the Synthetic Minority Oversampling TEchnique
 * (SMOTE). The original dataset must fit entirely in memory. The amount of
 * SMOTE and number of nearest neighbors may be specified. For more information,
 * see <br/>
 * <br/>
 * Nitesh V. Chawla et. al. (2002). Synthetic Minority Over-sampling Technique.
 * Journal of Artificial Intelligence Research. 16:321-357.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{al.2002,
 *    author = {Nitesh V. Chawla et. al.},
 *    journal = {Journal of Artificial Intelligence Research},
 *    pages = {321-357},
 *    title = {Synthetic Minority Over-sampling Technique},
 *    volume = {16},
 *    year = {2002}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start -->
 * Valid options are:
 * <p/>
 *
 * <pre> -S &lt;num&gt;
 *  Specifies the random number seed (default 1)</pre>
 *
 * <pre> -P &lt;percentage&gt;
 *  Specifies percentage of SMOTE instances to create. (default 100.0)
 * </pre>
 *
 * <pre> -K &lt;nearest-neighbors&gt;
 *  Specifies the number of nearest neighbors to use. (default 5)
 * </pre>
 *
 * <pre> -C &lt;value-index&gt;
 *  Specifies the index of the nominal class value to SMOTE
 *  (default 0: auto-detect non-empty minority class))
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Ryan Lichtenwalter (rlichtenwalter@gmail.com)
 * @version $Revision: 1.0$
 */
public class CMEOT extends Filter
        implements SupervisedFilter, OptionHandler, TechnicalInformationHandler {

    static final long serialVersionUID = -1653880819059250364L;

    protected int m_NearestNeighbors = 5;
    protected int m_RandomSeed = 1;
    protected double m_Percentage = 100.0;
    protected String m_ClassValueIndex = "0";
    protected boolean m_DetectMinorityClass = true;

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {
        return "Resamples a dataset by applying the Synthetic Minority Oversampling TEchnique (SMOTE)."
                + " The original dataset must fit entirely in memory."
                + " The amount of SMOTE and number of nearest neighbors may be specified."
                + " For more information, see \n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);

        result.setValue(Field.AUTHOR, "Nitesh V. Chawla et. al.");
        result.setValue(Field.TITLE, "Synthetic Minority Over-sampling Technique");
        result.setValue(Field.JOURNAL, "Journal of Artificial Intelligence Research");
        result.setValue(Field.YEAR, "2002");
        result.setValue(Field.VOLUME, "16");
        result.setValue(Field.PAGES, "321-357");

        return result;
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1.0$");
    }

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // attributes
        result.enableAllAttributes();
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        return result;
    }

    /**
     * Default constructor.
     */
    public CMEOT() {
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {
        Vector newVector = new Vector(3);
        newVector.addElement(new Option(
                "\tSpecifies the random number seed (default 1)",
                "S", 1, "-S <num>"));
        newVector.addElement(new Option(
                "\tSpecifies percentage of SMOTE instances to create. (default 100.0)\n",
                "P", 1, "-P <percentage>"));
        newVector.addElement(new Option(
                "\tSpecifies the number of nearest neighbors to use. (default 5)\n",
                "K", 1, "-K <nearest-neighbors>"));
        newVector.addElement(new Option(
                "\tSpecifies the index of the nominal class value to SMOTE\n"
                + "\t(default 0: auto-detect non-empty minority class))\n",
                "C", 1, "-C <value-index>"));
        /*		newVector.addElement( new Option(
         "\tSpecifies whether the non-empty minority class should be auto-detected (default true)\n",
         "d", 1, "-d" ) ); */

        return newVector.elements();
    }

    /**
     * Parses a given list of options.
     *
     * <!-- options-start -->
     * Valid options are:
     * <p/>
     *
     * <pre> -S &lt;num&gt;
     *  Specifies the random number seed (default 1)</pre>
     *
     * <pre> -P &lt;percentage&gt;
     *  Specifies percentage of SMOTE instances to create. (default 100.0)
     * </pre>
     *
     * <pre> -K &lt;nearest-neighbors&gt;
     *  Specifies the number of nearest neighbors to use. (default 5)
     * </pre>
     *
     * <pre> -C &lt;value-index&gt;
     *  Specifies the index of the nominal class value to SMOTE
     *  (default 0: auto-detect non-empty minority class))
     * </pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        String seedStr = Utils.getOption('S', options);
        if (seedStr.length() != 0) {
            setRandomSeed(Integer.parseInt(seedStr));
        } else {
            setRandomSeed(1);
        }

        String percentageStr = Utils.getOption('P', options);
        if (percentageStr.length() != 0) {
            double percentage = new Double(percentageStr).doubleValue();
            if (percentage < 0) {
                throw new Exception("P must be >= 0");
            } else {
                setPercentage(percentage);
            }
        } else {
            setPercentage(100.0);
        }

        String nnStr = Utils.getOption('K', options);
        if (nnStr.length() != 0) {
            int nn = Integer.parseInt(nnStr);
            if (nn < 1) {
                throw new Exception("K must be >= 1");
            } else {
                setNearestNeighbors(nn);
            }
        } else {
            setNearestNeighbors(5);
        }
        String avgStr = Utils.getOption('A', options);
        if (avgStr.length() != 0) {
            int nn = Integer.parseInt(avgStr);
            if (nn <= 1) {
                if (nn == 1) {
                    avgflag = true;
                } else {
                    avgflag = false;
                }
            } else {
                throw new Exception("A must be 0,1");

            }
        } else {
            avgflag = true;
        }

        //		setDetectMinorityClass( Utils.getFlag( 'd', options ) );
        String classValueIndexStr = Utils.getOption('C', options);
        if (classValueIndexStr.length() != 0) {
            setClassValue(classValueIndexStr);
            //			int classValueIndex = Integer.parseInt( classValueIndexStr );

            if (classValueIndexStr.equals("0")) {
                m_DetectMinorityClass = true;
                /*                        } else if( classValueIndex >= getInputFormat().numClasses() ) {
                 throw new Exception( "value index must be < the number of classes" ); */
            } else {
                //setClassValue( classValueIndex );
                m_DetectMinorityClass = false;
            }
        } else {
            m_DetectMinorityClass = true;
        }
    }

    /**
     * Gets the current settings of the filter.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        String[] options = new String[11];
        int current = 0;
        options[current++] = "-C";
        options[current++] = getClassValue();
        /*		if( getDetectMinorityClass() ) {
         options[current++] = "-d";
         } */
        options[current++] = "-K";
        options[current++] = "" + getNearestNeighbors();
        options[current++] = "-P";
        options[current++] = "" + getPercentage();
        options[current++] = "-S";
        options[current++] = "" + getRandomSeed();

        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String randomSeedTipText() {
        return "The seed used for random sampling.";
    }

    /**
     * Gets the random number seed.
     *
     * @return the random number seed.
     */
    public int getRandomSeed() {

        return m_RandomSeed;
    }

    /**
     * Sets the random number seed.
     *
     * @param newSeed the new random number seed.
     */
    public void setRandomSeed(int newSeed) {

        m_RandomSeed = newSeed;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String percentageTipText() {
        return "The percentage of SMOTE instances to create.";
    }

    /**
     * Sets the percentage of SMOTE instances to create.
     *
     * @param percentage
     */
    public void setPercentage(double percentage) {
        m_Percentage = percentage;
    }

    /**
     * Gets the percentage of SMOTE instances to create.
     *
     * @return the percentage of SMOTE instances to create
     */
    public double getPercentage() {
        return m_Percentage;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String nearestNeighborsTipText() {
        return "The number of nearest neighbors to use.";
    }

    /**
     * Sets the number of nearest neighbors to use.
     *
     * @param nearestNeighbors
     */
    public void setNearestNeighbors(int nearestNeighbors) {
        m_NearestNeighbors = nearestNeighbors;
    }

    /**
     * Gets the number of nearest neighbors to use.
     *
     * @return the number of nearest neighbors to use
     */
    public int getNearestNeighbors() {
        return m_NearestNeighbors;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String classValueTipText() {
        return "The index of the class value to which SMOTE should be applied. "
                + "Use a value of 0 to auto-detect the non-empty minority class.";
    }

    /**
     * Sets the index of the class value to which SMOTE should be applied.
     *
     * @param classValueIndex
     */
    public void setClassValue(String classValueIndex) {
        m_ClassValueIndex = classValueIndex;
    }

    /**
     * Gets the index of the class value to which SMOTE should be applied.
     *
     * @return the index of the clas value to which SMOTE should be applied
     */
    public String getClassValue() {
        return m_ClassValueIndex;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String detectMinorityClassTipText() {
        return "Whether the non-empty minority class should be automatically "
                + "detected. When this is true, classValueIndex has no meaning.";
    }

    /**
     * Sets whether or not the minority class should be auto-detected.
     *
     * @param detectMinorityClass
     */
    public void setDetectMinorityClass(boolean detectMinorityClass) {
        m_DetectMinorityClass = detectMinorityClass;
    }

    /**
     * Gets whether or not the minority class should be auto-detected.
     *
     * @return true if the minority class should be auto-detected; false
     * otherwise
     *
     * public boolean getDetectMinorityClass() { return m_DetectMinorityClass; }
     */
    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo an Instances object containing the input instance
     * structure (any instances contained in the object are ignored - only the
     * structure is required).
     * @return true if the outputFormat may be collected immediately
     * @exception Exception if the input format can't be set successfully
     */
    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        super.setInputFormat(instanceInfo);
        super.setOutputFormat(instanceInfo);
        return true;
    }

    /**
     * Input an instance for filtering. Filter requires all training instances
     * be read before producing output.
     *
     * @param instance the input instance
     * @return true if the filtered instance may now be collected with output().
     * @exception IllegalStateException if no input structure has been defined
     */
    public boolean input(Instance instance) {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }
        if (m_NewBatch) {
            resetQueue();
            m_NewBatch = false;
        }
        if (m_FirstBatchDone) {
            push(instance);
            return true;
        } else {
            bufferInput(instance);
            return false;
        }
    }

    /**
     * Signify that this batch of input to the filter is finished. If the filter
     * requires all instances prior to filtering, output() may now be called to
     * retrieve the filtered instances.
     *
     * @return true if there are instances pending output
     * @exception IllegalStateException if no input structure has been defined
     * @exception Exception if provided options cannot be executed on input
     * instances
     */
    public boolean batchFinished() throws Exception {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        if (!m_FirstBatchDone) {
            // Do SMOTE, and clear the input instances.
            filterapply = doCME();
        }
        flushInput();

        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }
    Random seedval = new Random(545656);

    /**
     * The procedure implementing the SMOTE algorithm. The output instances are
     * pushed onto the output queue for collection.
     *
     * @exception Exception if provided options cannot be executed on input
     * instances
     */
    public boolean doCME() throws Exception {
        int minIndex = 0;
        int min = Integer.MAX_VALUE;
        int maxcount = 0;

//        ArrayList<Integer> minalreadyindex = new ArrayList();
//        for (int i = 0; i < classindex.size(); i++) {
//            minalreadyindex.add(classindex.get(i).index);
//        }
        if (m_DetectMinorityClass) {
            // find minority class
            int[] classCounts = getInputFormat().attributeStats(getInputFormat().classIndex()).nominalCounts;
            for (int i = 0; i < classCounts.length; i++) {
                if (i == minclassindex) {
                    min = classCounts[i];
                    minIndex = i;
                }
                if (classCounts[i] != 0 && classCounts[i] > maxcount) {
                    maxcount = classCounts[i];
                    //    minIndex = i;
                }
            }
        }

        // compose minority class dataset
        // also push all dataset instances
        Instances sample = getInputFormat().stringFreeStructure();

        Instances nearknn = getInputFormat().stringFreeStructure();
        Instances orignalset = getInputFormat().stringFreeStructure();

        Enumeration instanceEnum = getInputFormat().enumerateInstances();

////        // compute Value Distance Metric matrices for nominal features
        // use this random source for all required randomness
        orignalset = getInputFormat().stringFreeStructure();
        sample = getInputFormat().stringFreeStructure();

        while (instanceEnum.hasMoreElements()) {
            Instance instance = (Instance) instanceEnum.nextElement();

            if (instance.classValue() == minclassindex) {
                push((Instance) instance.copy());

                orignalset.add((Instance) instance.copy());
                //  sample.add((Instance) instance.copy());
            }
        }
        int m1 = (int) (maxcount * 0.40);

        System.out.print("\nFilter Max Count :" + maxcount + ", Min Count : " + min + ":" + !(m1 < min));
        if (m1 < min  || min == 0) {
            return false;
        }
        if (option == 2) {
            maxcount = 0;
            if (m_DetectMinorityClass) {
                // find minority class
                int[] classCounts = getInputFormat().attributeStats(getInputFormat().classIndex()).nominalCounts;
                for (int i = 0; i < classCounts.length; i++) {
                    if (i != minIndex) {
                        maxcount += classCounts[i];
                    }

                }
            }
        }
     //   index = minIndex;

        maxcount = ((maxcount * 40) / 60) - min;
        System.out.println(" Max xount : " + maxcount);
        int nearestNeighbors;
        if (min <= getNearestNeighbors()) {
            nearestNeighbors = min - 1;
        } else {
            nearestNeighbors = getNearestNeighbors();
        }
        int totalos = maxcount / min + 1;

        Instances ret = process(maxcount, min, minIndex, nearestNeighbors, sample, sample);
        for (int i = 0; i < ret.numInstances(); i++) {
            Instance instanceI = ret.get(i);
            double[] d = instanceI.toDoubleArray();
            instanceI = (Instance) sample.get(0).copy();
            for (int k = 0; k < instanceI.numAttributes(); k++) {
                instanceI.setValue(k, d[k]);
            }
            instanceI.setDataset(getInputFormat());
            push(instanceI);
            //  ret.add(instanceI);
        }

        return true;
    }

    public Instances process(int maxcount, int min, int minIndex, int nearestNeighbors, Instances orignalset, Instances sample) {
        ArrayList<ArrayList< Instance>> alInstances = new ArrayList();
        Random rand = new Random(getRandomSeed());
        Instances nearknn = orignalset.stringFreeStructure();

        Map vdmMap = new HashMap();
        Enumeration attrEnum = orignalset.enumerateAttributes();
        while (attrEnum.hasMoreElements()) {
            Attribute attr = (Attribute) attrEnum.nextElement();
            if (!attr.equals(orignalset.classAttribute())) {
                if (attr.isNominal() || attr.isString()) {
                    double[][] vdm = new double[attr.numValues()][attr.numValues()];
                    vdmMap.put(attr, vdm);
                    int[] featureValueCounts = new int[attr.numValues()];
                    int[][] featureValueCountsByClass = new int[orignalset.classAttribute().numValues()][attr.numValues()];
                    Enumeration<Instance> instanceEnum = orignalset.enumerateInstances();
                    while (instanceEnum.hasMoreElements()) {
                        Instance instance = (Instance) instanceEnum.nextElement();
                        int value = (int) instance.value(attr);
                        int classValue = (int) instance.classValue();
                        featureValueCounts[value]++;
                        featureValueCountsByClass[classValue][value]++;
                    }
                    for (int valueIndex1 = 0; valueIndex1 < attr.numValues(); valueIndex1++) {
                        for (int valueIndex2 = 0; valueIndex2 < attr.numValues(); valueIndex2++) {
                            double sum = 0;
                            for (int classValueIndex = 0; classValueIndex < orignalset.numClasses(); classValueIndex++) {
                                double c1i = (double) featureValueCountsByClass[classValueIndex][valueIndex1];
                                double c2i = (double) featureValueCountsByClass[classValueIndex][valueIndex2];
                                double c1 = (double) featureValueCounts[valueIndex1];
                                double c2 = (double) featureValueCounts[valueIndex2];
                                double term1 = c1i / c1;
                                double term2 = c2i / c2;
                                sum += Math.abs(term1 - term2);
                            }
                            vdm[valueIndex1][valueIndex2] = sum;
                        }
                    }
                }
            }
        }
        int os = 0, pos = -1;
        int totalcount = 0, ptotalcount = -1, pmaxcount = maxcount, oscount = 0, iter = 1000, end = 0;

        long initialseedValue = rand.nextLong();
        while (maxcount > 0 && end < 100) {
            os = totalcount / min;

            ptotalcount = totalcount;
            //   if (os > pos) {
            pos = os;
            if (nearknn == null || nearknn.size() == 0) {
                nearknn = orignalset.stringFreeStructure();
                for (int i = 0; i < orignalset.size(); i++) {
                    nearknn.add((Instance) orignalset.get(i).copy());

                }
            }
            //   if (os > 0) {
            for (int j = oscount; j < sample.size(); j++) {
                nearknn.add((Instance) sample.get(j).copy());
                // sample.add((Instance) sample.get(j).copy());
            }
            oscount = sample.size();

            boolean flag = false;
            while (!flag && maxcount > 0 && end < 100) {

                if (pmaxcount == maxcount) {
                    end++;
                } else {
                    end = 0;
                }
                pmaxcount = maxcount;

                try {
                    SimpleKMeans kMeans = new SimpleKMeans();
                    kMeans.setNumClusters(nearestNeighbors);
                    iter++;
                    kMeans.setSeed(seedval.nextInt());
                    kMeans.buildClusterer(nearknn);

                    // print out the cluster centroids
                    Instances centroids = kMeans.getClusterCentroids();

                    for (int i = 0; i < centroids.numInstances(); i++) {

                        if (!chkDublicate(sample, orignalset, centroids.instance(i))) {
                            centroids.instance(i).setDataset(orignalset.stringFreeStructure());
                            centroids.instance(i).setClassValue(minIndex);
                            sample.add(centroids.instance(i));
                            flag = true;
                            maxcount--;
                            totalcount++;
                            if (maxcount == 0) {
                                break;
                            }
                        }

                    }
                } catch (Exception ex) {
                    Logger.getLogger(CMEOT.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            os++;
        }

        return sample;
    }

    public boolean chkDublicate(Instances al, Instances al1,
            Instance synthetic) {
        boolean flag = false;

        // val = (alInstances.get(i).size() / (os + 1));
        for (int j = 0; j < al.size(); j++) {
            if (al.get(j).equals(synthetic)) {
                return true;
            }
        }

        for (int j = 0; j < al1.size(); j++) {
            if (al1.get(j).equals(synthetic)) {
                return true;
            }
        }

        return flag;
    }

}
