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
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Vector;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
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
public class MMCBUOST extends Filter
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
    public MMCBUOST() {
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
            filterapply = doMMCBUOST();
        }
        flushInput();

        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }

    /**
     * The procedure implementing the SMOTE algorithm. The output instances are
     * pushed onto the output queue for collection.
     *
     * @exception Exception if provided options cannot be executed on input
     * instances
     */
    public boolean doMMCBUOST() throws Exception {
        Random seedval = new Random(545656);

        int minIndex = 0;
        int maxIndex = 0;

        int min = minclassindex;
        int maxcount = 0;

//        ArrayList<Integer> minalreadyindex = new ArrayList();
//        for (int i = 0; i < Common.classindex.size(); i++) {
//            minalreadyindex.add(Common.classindex.get(i).index);
//        }
        if (m_DetectMinorityClass) {
            // find minority class
            int[] classCounts = getInputFormat().attributeStats(getInputFormat().classIndex()).nominalCounts;
            for (int i = 0; i < classCounts.length; i++) {
                //   if ((option == 0 || (option == 1 && minalreadyindex.indexOf(i) == -1) || (option == 2 && minalreadyindex.indexOf(i) == -1)) && classCounts[i] != 0 && classCounts[i] < min) {
                if (minclassindex == i) {
                    min = classCounts[i];
                    minIndex = i;
                }
                if (classCounts[i] != 0 && classCounts[i] > maxcount) {
                    maxcount = classCounts[i];
                    maxIndex = i;
                }
            }
        }

        // compose minority class dataset
        // also push all dataset instances
        Instances sample = getInputFormat().stringFreeStructure();

        Instances nearknn = getInputFormat().stringFreeStructure();
        Instances orignalset = getInputFormat().stringFreeStructure();
        Instances dataset = getInputFormat().stringFreeStructure();

        // use this random source for all required randomness
        Random rand = new Random(getRandomSeed());
        dataset = getInputFormat().stringFreeStructure();
        orignalset = getInputFormat().stringFreeStructure();
        sample = getInputFormat().stringFreeStructure();
        Enumeration<Instance> instanceEnum = getInputFormat().enumerateInstances();
        while (instanceEnum.hasMoreElements()) {
            Instance instance = (Instance) instanceEnum.nextElement();
            if (instance.classValue() == minIndex || (instance.classValue() == maxIndex || option == 2)) {
                dataset.add(instance);
                if (instance.classValue() == minclassindex) {
                    push((Instance) instance.copy());

                    sample.add(instance);
                    //  sample.add((Instance) instance.copy());
                } else {
                    orignalset.add(instance);
                }
            }
        }
        int m1 = (int) (maxcount * 0.40);
        if (m1 < min || min == 0) {
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

        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(getNearestNeighbors());
        kMeans.setSeed(seedval.nextInt());
        kMeans.buildClusterer(sample);
        Instances centroids = kMeans.getClusterCentroids();
        int max = 0;
        System.out.print("\n Minorty Cluser (" + sample.size() + ") : ");
        ArrayList<Integer> cntal1 = new ArrayList();
        ArrayList<Instances> data = new ArrayList();
        for (int i = 0; i < centroids.size(); i++) {
            cntal1.add(0);
            data.add(new Instances(getInputFormat().stringFreeStructure()));
        }
        for (int i = 0; i < sample.size(); i++) {
            int index = kMeans.clusterInstance(sample.instance(i));
            cntal1.set(index, cntal1.get(index) + 1);
            data.get(index).add(sample.instance(i));
        }
        for (int i = 0; i < centroids.size(); i++) {
            if (max < cntal1.get(i)) {
                max = cntal1.get(i);
            }
            System.out.print("\t" + cntal1.get(i));
        }

        SimpleKMeans kMeans_orignalset = new SimpleKMeans();
        kMeans_orignalset.setNumClusters(getNearestNeighbors());
        kMeans_orignalset.setSeed(seedval.nextInt());
        kMeans_orignalset.buildClusterer(orignalset);
        Instances centroids1 = kMeans_orignalset.getClusterCentroids();
        ArrayList<Integer> cntal = new ArrayList();
        ArrayList<Instances> data1 = new ArrayList();
        for (int i = 0; i < centroids1.size(); i++) {
            cntal.add(0);
            data1.add(new Instances(getInputFormat().stringFreeStructure()));
        }
        for (int i = 0; i < orignalset.size(); i++) {
            int index = kMeans_orignalset.clusterInstance(orignalset.instance(i));
            cntal.set(index, cntal.get(index) + 1);
            data1.get(index).add(orignalset.instance(i));
// cntal.add(centroids1.get(i).dataset().size());
        }
        Collections.sort(cntal);
        System.out.print("\n MAjority Cluser : ");
        for (int i = 0; i < cntal.size(); i++) {
            System.out.print("\t" + cntal.get(i));
        }
        double max_orginalset = 0;

        for (int i = 0; i < cntal.size(); i++) {

            max_orginalset += cntal.get(i);

        }
        max_orginalset = max_orginalset / cntal.size();

        for (int i = 0; i < centroids1.size(); i++) {
            ArrayList<SafeLevel> al = new ArrayList();
            for (int j = 0; j < data1.get(i).numInstances(); j++) {
                al.add(new SafeLevel(data1.get(i).instance(j),
                        getSafeLevel(dataset, data1.get(i).instance(j), getNearestNeighbors(), minIndex)));
            }
            Collections.sort(al);
            for (int j = data1.get(i).numInstances() - 1; j >= max_orginalset && j < data1.get(i).numInstances(); j--) {
                data1.get(i).delete(j);
            }
        }
        System.out.print("\nRemove MAjority Cluser : ");
        for (int i = 0; i < cntal.size(); i++) {
            System.out.print("\t" + data1.get(i).numInstances());
        }

        orignalset = getInputFormat().stringFreeStructure();
        for (int i = 0; i < centroids1.size(); i++) {
            for (int j = 0; j < data1.get(i).numInstances(); j++) {
                orignalset.add(data1.get(i).instance(j));
            }

        }

        maxcount = orignalset.size();

        maxcount = (((maxcount * 40) / 60) - min);

        System.out.print("\nFilter Max Count :" + maxcount + ", Min Count : " + min + ":" + !(m1 < min));
        Instances sample_data = data.get(0);
        Instances d1 = null;
        if (mmsalgo == 2) {
            for (int i = 1; i < centroids.size(); i++) {
                sample_data.addAll(0, data.get(i));
            }
            Filter filter[] = new Filter[]{new MEMMOT(), new MMMmOT(), new CMEOT(), new SMOTE(), new NFNMOT()};
            int k1 = sample_data.size();
            d1 = ((Filter) Common.filter[Common.mmmcnousalgo].getClass().newInstance()).process(maxcount, sample_data.size(), minIndex,
                    getNearestNeighbors(), orignalset, sample_data);
            System.out.println("\nCluster " + 0 + " : " + k1 + ":" + d1.size());
            for (int j = 0; j < d1.size(); j++) {
                Instance instanceI = d1.get(j);
                double[] d = instanceI.toDoubleArray();
                instanceI = (Instance) sample_data.get(0).copy();
                for (int k = 0; k < instanceI.numAttributes(); k++) {
                    instanceI.setValue(k, d[k]);
                }
                instanceI.setDataset(getInputFormat());

                push(instanceI);
            }

        } else {

            int centcount = maxcount / centroids.size();
            for (int i = 0; i < centroids.size(); i++) {
                sample_data = data.get(i);
                if (sample_data.numInstances() > 0) {
                    int k1 = data.get(i).numInstances();

                    {
                        d1 = ((Filter) Common.filter[Common.mmmcnousalgo].getClass().newInstance()).process(centcount, sample_data.size(), minIndex,
                                getNearestNeighbors(), orignalset, sample_data);
                    }
                    System.out.println("\nCluster " + i + " : " + k1 + ":" + d1.size());
                    for (int j = 0; j < d1.size(); j++) {
                        Instance instanceI = d1.get(j);
                        double[] d = instanceI.toDoubleArray();
                        instanceI = (Instance) sample_data.get(0).copy();
                        for (int k = 0; k < instanceI.numAttributes(); k++) {
                            instanceI.setValue(k, d[k]);
                        }
                        instanceI.setDataset(getInputFormat());

                        push(instanceI);
                    }
                }
            }
        }
        return true;
    }

    public int getSafeLevel(Instances nearknn, Instance instanceI, int localnearestNeighbors, int minIndex) {

        //      for (int i = 0; i < sample.numInstances(); i++) {
        //        Instance instanceI = sample.instance(i);
        // find k nearest neighbors for each instance
        List<Object[]> distanceToInstance = new LinkedList();
        for (int j = 0; j < nearknn.numInstances(); j++) {
            Instance instanceJ = nearknn.instance(j);
            double distance = 0;

            //if (distval[j][i] == -1) {
            Enumeration<Attribute> attrEnum = nearknn.enumerateAttributes();
            while (attrEnum.hasMoreElements()) {
                Attribute attr = (Attribute) attrEnum.nextElement();
                if (!attr.equals(nearknn.classAttribute())) {
                    double iVal = instanceI.value(attr);
                    double jVal = instanceJ.value(attr);
                    if (attr.isNumeric()) {
                        distance += Math.pow(iVal - jVal, 2);
                    }
                }
            }
            distance = Math.pow(distance, .5);
            distanceToInstance.add(new Object[]{distance, instanceJ});
        }

        if (distanceToInstance.size() > 0) {
            // sort the neighbors according to distance
            try {
                Collections.sort(distanceToInstance, new Comparator() {
                    public int compare(Object o1, Object o2) {
                        try {
                            double distance1 = (Double) ((Object[]) o1)[0];
                            double distance2 = (Double) ((Object[]) o2)[0];
                            return Double.compare(distance1, distance2);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        return 0;
                    }
                });
            } catch (Exception e) {
                e.printStackTrace();
            }

        }

        int minority = 0;
        for (int i = 0; i < distanceToInstance.size(); i++) {
            // val = (alInstances.get(i).size() / (os + 1));
            if (((Instance) distanceToInstance.get(i)[1]).classValue() == minIndex) {
                minority++;
            }

        }
        return minority;

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
