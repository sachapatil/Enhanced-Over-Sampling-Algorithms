/*s
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eost;

import algo.Common;
import com.sun.media.parser.audio.AiffParser;
import static eost.CreateHadoopDataset.INPUT_PATH;
import static eost.CreateHadoopDataset.host;
import static eost.EOST.print;
import static eost.EOST.testData;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import javassist.bytecode.stackmap.BasicBlock;
import javax.imageio.ImageIO;
import javax.imageio.stream.ImageInputStream;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.NLineInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;

public class EOSTMapReduce extends Configured implements Tool {

    public static class NLineMapper extends Mapper<LongWritable, Text, Text, Text> {

        int option;
        int mmsalgo;
        int filter;
        int kmean;
        String inputFile = "";

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context); //To change body of generated methods, choose Tools | Templates.
            this.option = context.getConfiguration().getInt("option", 0);
            this.filter = context.getConfiguration().getInt("filter", 0);
            this.mmsalgo = context.getConfiguration().getInt("mmsalgo", 0);
            this.kmean = context.getConfiguration().getInt("kmean", 0);
            this.inputFile = context.getConfiguration().getStrings("inputfile")[0];
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            int minindex = Integer.parseInt(value.toString());
            String out = "";
            FileSystem hdfs = FileSystem.get(context.getConfiguration());
            try {
                FSDataInputStream filein = hdfs.open(new Path("/dfs/eost/dataset/" + inputFile));
                Instances loader = new ConverterUtils.DataSource(filein).getDataSet();
                Filter f = Common.filter[option];

                out = "/dfs/eost/output/" + minindex + inputFile.split("/")[1];

                boolean filterappply = Filter.filterFile(f,
                        new String[]{"-i", "/dfs/eost/dataset/" + inputFile, "-o", out, "-c", "last",
                            "-decimal", "0", "-K",
                            String.valueOf(kmean), "-A", "0",
                            "-option",
                            String.valueOf(option), "-minindex",
                            String.valueOf(minindex), "-mmsalgo", String.valueOf(mmsalgo)}, hdfs);

            } catch (Exception ex) {
                Logger.getLogger(EOSTMapReduce.class.getName()).log(Level.SEVERE, null, ex);
            }

            context.write(new Text("out"), new Text(out + "##" + minindex));
        }
    }

    public static void loadFile(final String file, final int m,
            final int kmean, final int option,
            final int mmsalgo, final int folds, final Classifier classifier) {
        try {
            {
                String name = new Path("/dfs/eost/dataset/" + file).getName();
                name = file.replaceAll(name, "cloud_" + name);
                DeleteAll.run("/dfs/eost/output/" + name);
            }
            UserGroupInformation ugi
                    = UserGroupInformation.createRemoteUser("hdfs");

            ugi.doAs(new PrivilegedExceptionAction<Void>() {

                public Void run() throws Exception {

                    try {

                        //1. Get the instance of COnfiguration
                        Configuration conf = new Configuration();
                        conf.set("fs.defaultFS", "hdfs://" + host + ":8020/dfs/nn");
                        conf.set("hadoop.job.ugi", "root");
                        FileSystem hdfs = FileSystem.get(conf);

                        if (!hdfs.isDirectory(new Path("/dfs/eost/lib"))) {
                            // for (int i = 0; i < f1.length; i++) {
                            hdfs.copyFromLocalFile(new Path("dist/lib"), new Path("/dfs/eost/lib"));

                            //  }
                            hdfs.copyFromLocalFile(new Path("dist/EOSTCloud.jar"), new Path("/dfs/eost/lib/EOSTCloud.jar"));

                        }
                        File f1[] = new File("dist").listFiles();
                        FileStatus[] dir = hdfs.listStatus(new Path("/dfs/eost/lib"));
                        Path[] libfile = new Path[dir.length];

                        for (int i = 0; i < dir.length; i++) {
                            libfile[i] = dir[i].getPath();//.toString();
                        }

                        conf.setInt(NLineInputFormat.LINES_PER_MAP, 1000);
                        conf.setInt("option", option);
                        conf.setInt("mmsalgo", mmsalgo);
                        conf.setInt("filter", m);
                        conf.setInt("kmean", kmean);

                        conf.setStrings("inputfile", file);

                        Job job = new Job(conf, "NLine Input Format");
                        job.setJarByClass(EOSTMapReduce.class);

                        job.setMapperClass(NLineMapper.class);
                        job.setReducerClass(NLineReducer.class);
                        job.setInputFormatClass(NLineInputFormat.class);

                        String name = new Path("/dfs/eost/dataset/" + file).getName();
                        name = file.replaceAll(name, "cloud_" + name);
                        FileInputFormat.addInputPath(job, new Path("/dfs/eost/dataset/" + name));
                        FileOutputFormat.setOutputPath(job, new Path("/dfs/eost/output/" + name));

                        job.setOutputKeyClass(Text.class);
                        job.setOutputValueClass(Text.class);

                        for (int ii = 0; ii < libfile.length; ii++) {
                            //=dir[i].getPath().toString();
                            //            System.out.println(libfile[ii].getName());
                            DistributedCache.addFileToClassPath(new Path("/dfs/eost/lib/" + libfile[ii].getName()), job.getConfiguration());

                        }

                        job.waitForCompletion(true);

                        System.out.println("-------------------------Original Data File " + new Path("/dfs/eost/dataset/" + file).getName() + ",folds:" + folds + "----------------------");
                        print(hdfs.open(new Path("/dfs/eost/dataset/" + file)));

                        testData(hdfs.open(new Path("/dfs/eost/dataset/" + file)), folds, classifier);

                        System.out.println("-------------------------Filter Data File (Algo:" + Common.filter[m].getClass().getSimpleName()
                                + ",Classifier:" + classifier.getClass().getSimpleName()
                                + ",k:" + kmean + ") Train File : " + new Path("/dfs/eost/output/" + file).getName()
                                + " folds :" + folds + "----------------------");
                        print(hdfs.open(new Path("/dfs/eost/output/" + file)));
                        testData(hdfs.open(new Path("/dfs/eost/output/" + file)), folds, classifier);
                        //hdfs.deleteOnExit(new Path("/dfs/eost/output/" + name));

                        hdfs.close();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    return null;
                }

            });
        } catch (Exception e) {

        }
    }

    @Override
    public int run(String[] strings) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String[] args) {
        MultilayerPerceptron mp = new MultilayerPerceptron();
        try {
            mp.setOptions(new String[]{"-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "a"});
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        for (int i = 0; i < Common.filter.length; i++) {
            int algo = i;
            loadFile("glass/glass.arff", 2, 5, algo, 2, 10, mp);
        }
//        int exitcode = ToolRunner.run(new EOSTMapReduce(), new String[]{"imagedb.txt", "imagedboutput1.txt"});
//        System.exit(exitcode);
    }

    public static class NLineReducer extends Reducer<Text, Text, Text, Text> {

        int option;
        int mmsalgo;
        int filter;
        int kmean;
        String inputFile = "";

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context); //To change body of generated methods, choose Tools | Templates.
            this.option = context.getConfiguration().getInt("option", 0);
            this.filter = context.getConfiguration().getInt("filter", 0);
            this.mmsalgo = context.getConfiguration().getInt("mmsalgo", 0);
            this.kmean = context.getConfiguration().getInt("kmean", 0);
            this.inputFile = context.getConfiguration().getStrings("inputfile")[0];

        }

        @Override
        public void reduce(Text key, Iterable<Text> value, Context context) throws IOException, InterruptedException {

            Configuration conf = new Configuration();
            conf.set("fs.defaultFS", "hdfs://hadoop1.example.com" + ":8020/dfs/nn");
            conf.set("hadoop.job.ugi", "root");

            //3. Get the HDFS instance
            FileSystem hdfs = FileSystem.get(conf);

            ArrayList<Common.InstanceIndex> classindex = new ArrayList();
            //     InputStream in = new InputStreamReader(filein);
            for (Iterator iterator = value.iterator(); iterator.hasNext();) {
                String str = iterator.next().toString();
                String out = str.split("##")[0];
                int minindex = Integer.parseInt(str.split("##")[1]);
                FSDataInputStream filein = hdfs.open(new Path(out));

                ArffLoader ar = new ArffLoader();
                ar.setSource(filein);
                classindex.add(new Common.InstanceIndex(ar.getDataSet(), minindex));

            }
            FSDataInputStream filein = hdfs.open(new Path("/dfs/eost/dataset/" + inputFile));

            ArffLoader ar = new ArffLoader();
            try {
                ar.setSource(filein);
                Instances ds = ar.getDataSet();
                ds.setClassIndex(ds.numAttributes() - 1);

                for (int k = ds.numInstances() - 1; k >= 0; k--) {
                    ds.remove(k);
                }

                for (int j = 0; j < classindex.size(); j++) {
                    for (int k = 0; k < classindex.get(j).in.numInstances(); k++) {
                        //                  if (classindex.get(j).index == (int) classindex.get(j).in.instance(k).classValue()) {
                        classindex.get(j).in.instance(k).setDataset(ds);
                        Instance in = classindex.get(j).in.instance(k);
                        ds.add(classindex.get(j).in.instance(k));
                        //                }
                    }
                }
                System.out.println("Number of Attribute : " + ds.numAttributes() + ": " + ds.classIndex());
                ds.setClassIndex(ds.numAttributes() - 1);
                ArffSaver ars = new ArffSaver();
                FSDataOutputStream fileout = hdfs.create(new Path("/dfs/eost/output/" + inputFile));

                ars.setDestination(fileout);
                ars.setInstances(ds);
                ars.writeBatch();

            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    /*    public static class NLineEmpInputFormat extends FileInputFormat<LongWritable,Text>{
     public static final String  LINES_PER_MAP = "mapreduce.input.lineinputformat.linespermap";
       
     public RecordReader<LongWritable, Text> getRecordReader(InputSplit split, TaskAttemptContext context) throws IOException{
     context.setStatus(split);
     return new LineRecordReader();
     }
     } */
}
