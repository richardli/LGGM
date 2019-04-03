package util;

import cern.jet.random.tdouble.Exponential;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import sampler.Latent_classifier;
import sampler.Latent_model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by zehangli on 12/10/16.
 */
public class ProcessVAdata {

    int P;
    int Ntrain;
    int N_test;
    double[][] data;
    double[][] data_test;
    double[][] Rinit;
    int[] cause;
    int[] cause_test;
    int G;
    double[][] delta;

    public static void main(String[] args) throws IOException {

        ProcessVAdata vadata = new ProcessVAdata();
        String prior = "expnew/typeK3";
        String train = "expnew/K_train0";
        String test = "expnew/K_test0";
        String directory = "/Users/zehangli/";
        int Nitr = 10;
        String name = "newK9-0-0-20F";
        String type = "SSSL";
        double sd0 = 0.5;
        boolean NB = false;
        boolean integrate = false;
        int PP = 92;
        int seed = 5432;
        int Ntrain = 10000;
        boolean update_with_test = true;
        boolean adaptive = true;
        double v0 = 0.01;
        double v1 = 1;
        double lambda = 10;
        double prob = 0.0001;
        boolean is_this_classification_job = true;
        boolean same_pop = false;
        boolean anneal = false;
        String nopenalty_file = "expnew/typeK_structure";
        double shrink = 1;
        double var0 = 1;
        String csmffile = "nofile";
        boolean Dirichlet = false;
        boolean savefull = true;
        String impossiblelist = "nofile";
        boolean hotstart = false;
        double a_sd0 = 0.001;
        double b_sd0 = 0.001;
        int burnin = (int) (Nitr / 2.0);
        int thin = 1;


        if(args.length > 0) {
            int counter = 0;
             prior = args[counter];
            counter++;
             train = args[counter];
            counter++;
             test = args[counter];
            counter++;
             directory = args[counter];
            counter++;
             Nitr = Integer.parseInt(args[counter]);
            counter++;
             name = args[counter];
            counter++;
             type = args[counter];
            counter++;
             sd0 = Double.parseDouble(args[counter]);
            counter++;
             NB = Boolean.parseBoolean(args[counter]);
            counter++;
             integrate = false;
            if (args.length > counter) integrate = Boolean.parseBoolean(args[counter]);
            counter++;
             PP = 0;
            if (args.length > counter) PP = Integer.parseInt(args[counter]);
            counter++;
             seed = 1;
            if (args.length > counter) seed = Integer.parseInt(args[counter]);
            counter++;
             Ntrain = Integer.MAX_VALUE;
            if (args.length > counter) Ntrain = Integer.parseInt(args[counter]);
            counter++;
             update_with_test = true;
            if (args.length > counter) update_with_test = Boolean.parseBoolean(args[counter]);
            counter++;
             adaptive = false;
            if (args.length > counter) adaptive = Boolean.parseBoolean(args[counter]);
            counter++;
             v0 = 0.001;
             v1 = 10;
             lambda = 1;
             prob = 1 / (PP + 0.0);
            if (args.length > counter) v0 = Double.parseDouble(args[counter]);
            counter++;
            if (args.length > counter) v1 = Double.parseDouble(args[counter]);
            counter++;
            if (args.length > counter) lambda = Double.parseDouble(args[counter]);
            counter++;
            if (args.length > counter) prob = Double.parseDouble(args[counter]);
            counter++;
             is_this_classification_job = true;
            if (args.length > counter) is_this_classification_job = Boolean.parseBoolean(args[counter]);
            counter++;
             same_pop = true;
            if (args.length > counter) same_pop = Boolean.parseBoolean(args[counter]);
            counter++;
             anneal = true;
            if (args.length > counter) anneal = Boolean.parseBoolean(args[counter]);
            counter++;
             nopenalty_file = "nofile";
            if (args.length > counter) nopenalty_file = args[counter];
            counter++;
             shrink = 1;
            if (args.length > counter) shrink = Double.parseDouble(args[counter]);
            counter++;
             var0 = 1;
            if (args.length > counter) var0 = Double.parseDouble(args[counter]);
            counter++;
            csmffile = "";
            if (args.length > counter) csmffile = args[counter];
            counter++;
             Dirichlet = false;
            if (args.length > counter) Dirichlet = Boolean.parseBoolean(args[counter]);
            counter++;
             savefull = false;
            if (args.length > counter) savefull = Boolean.parseBoolean(args[counter]);
            counter++;
            if (args.length > counter) burnin = Integer.parseInt(args[counter]);
            counter++;
            if (args.length > counter) thin = Integer.parseInt(args[counter]);
            counter++;
             impossiblelist = "nofile";
            if (args.length > counter) impossiblelist = args[counter];
            counter++;
            if (args.length > counter) hotstart = Boolean.parseBoolean(args[counter]);
            counter++;
            if (args.length > counter) a_sd0 = Double.parseDouble(args[counter]);
            counter++;
            if (args.length > counter) b_sd0 = Double.parseDouble(args[counter]);
            counter++;
            System.out.println(Arrays.toString(args));
        }

        int maxTest = Integer.MAX_VALUE;
        // if this is not to test classification, use the test set as if labels are known.
        if (!is_this_classification_job) {
            maxTest = 0;
            Ntrain = Integer.MAX_VALUE;
            update_with_test = false;
        }

        String dir0 = "../data/";
        if(directory.equals("/Users/zehangli/")) dir0 = "/Users/zehangli/Bitbucket/LatentGaussian/data/";
//        if(directory.equals("/Users/zehangli/Bitbucket-repos/LatentGaussian/data/test")) dir0 = "/Users/zehangli/Bitbucket-repos/LatentGaussian/data/";

        vadata.addData(dir0 + prior + "_delta.csv", dir0 + train + ".csv",PP, Ntrain, seed, Integer.MAX_VALUE);
        vadata.addTestData(dir0 + test + ".csv", PP, maxTest);

        // 1 : known edge, 0: no known edge
        double[][] penalty_mat = vadata.addgraph(dir0 + nopenalty_file + ".csv", PP, nopenalty_file.equals("nofile"));

        vadata.addinitialR("to-remove-this-initialization-place-holder", hotstart);


        // sd0 = 0.5;
        int N = vadata.Ntrain + vadata.N_test;
        int N_test = vadata.N_test;
        int P = vadata.P;
        int G = vadata.G;
        double power = 0;
        boolean informative_prior = true;
        if(csmffile.equals("nofile")) informative_prior = false;
        if(csmffile.equals("expnew/InterVAcsmf")) informative_prior = false;
        if(csmffile.equals("expnew/csmf")) informative_prior = true;

        boolean random_init = false;
        boolean update_sparsity = false;



        String expriment_name = name;
        String currentdir = directory + expriment_name + "/";
        String currentfile = expriment_name;

        Latent_classifier model = new Latent_classifier(Nitr, burnin, thin, N, N_test, P, G, type);
        model.Dirichlet = Dirichlet;


        model.data.init_adaptive(sd0, a_sd0, b_sd0, adaptive, power);
        model.data.N = N;
        model.data.G = vadata.G;
        model.data.ID = new String[model.data.N];
        model.data.data = new double[model.data.N][vadata.P];
        model.data.latent = new double[model.data.N][vadata.P];
        model.data.expanded = new double[model.data.N][vadata.P];
        model.data.Delta = new double[G][vadata.P];
        model.data.Delta_expanded = new double[G][vadata.P];
        model.savefull = savefull;

        // need to update this when cont data are included
        model.data.type = new int[vadata.P];
        for (int i = 0; i < vadata.P; i++) {
            model.data.type[i] = 1;
        }
        double DirichletAlpha = model.data.N;//10000;//(model.data.N - model.data.N_test) / 2.0;
        if(!informative_prior) DirichletAlpha = 1.0;

        if(csmffile.equals("expnew/InterVAcsmf")){
            double[] csmf = readPriorCSMF(dir0 + csmffile + ".csv");
            model.init_prior(G, N_test, DirichletAlpha, penalty_mat, var0, informative_prior, csmf);
        }else{
            Random rand = new Random(seed);
            model.init_prior(G, N_test, DirichletAlpha, rand, penalty_mat, var0, informative_prior);
        }
        model.update_with_test = update_with_test;
        Random rand = new Random(seed);

        for (int i = 0; i < N_test; i++) {
            for (int j = 0; j < P; j++) {
                model.data.data[i][j] = vadata.data_test[i][j];
            }
            model.data.membership[i] = G;
            model.data.ID[i] = "test" + i;
            model.test_case[i] = true;
        }

        model.keeptruth = true;
        model.true_membership = new int[model.N];
        for (int i = 0; i < N_test; i++) {
            model.true_membership[i] = vadata.cause_test[i];
        }


        for (int i = N_test; i < N; i++) {
            for (int j = 0; j < P; j++) {
                model.data.data[i][j] = vadata.data[i - N_test][j];
            }
            model.data.membership[i] = vadata.cause[i - N_test];
            model.data.ID[i] = "id" + i;
            model.test_case[i] = false;
        }
        for (int j = 0; j < P; j++) {
            for (int g = 0; g < G; g++) {
                model.data.mu[g][j] = vadata.delta[g][j];
//                model.data.mu[g][j] = Math.sqrt(Math.abs(vadata.delta[g][j])) * (vadata.delta[g][j]> 0 ? 1 : -1);
                model.data.Delta[g][j] =vadata.delta[g][j];
//                model.data.Delta[g][j] = Math.sqrt(Math.abs(vadata.delta[g][j])) * (vadata.delta[g][j] > 0 ? 1 : -1);
            }
            model.data.type[j] = 1;
        }

        if(csmffile.equals("nofile")  | csmffile.equals("expnew/InterVAcsmf")){
            model.update_prior_prob(rand);
        }else{
            double[] csmf = readPriorCSMF(dir0 + csmffile + ".csv");
            model.update_prior_prob(csmf);
        }

        MersenneTwister rngEngin = new MersenneTwister(seed);
        DoubleMersenneTwister rngEngin2 = new DoubleMersenneTwister(seed);

        NormalDistribution rngN = new NormalDistribution(rngEngin, 0, 1);
        Exponential rngE = new Exponential(1, rngEngin2);
        Gamma rngG = new Gamma(1.0, 1.0, rngEngin2);
        if(type.equals("PX")){
            model.update_variable_type(model.cov_px);
        }else if(type.equals("SSSL")){
            model.update_variable_type(model.cov_sssl);
        }
        model.findimpossible();

        model.initial_test_label(rand, rngG, rngN, rngE, same_pop, random_init);
        model.data.update_groupcounts(model.update_with_test);

        for (int j = 0; j < P; j++) {
            for (int g = 0; g < G; g++) {
                model.data.mu[g][j] /= shrink;
//                model.data.mu[g][j] = Math.sqrt(Math.abs(vadata.delta[g][j])) * (vadata.delta[g][j]> 0 ? 1 : -1);
                model.data.Delta[g][j]/= shrink;
//                model.data.Delta[g][j] = Math.sqrt(Math.abs(vadata.delta[g][j])) * (vadata.delta[g][j] > 0 ? 1 : -1);
            }
            model.data.type[j] = 1;
        }
        model.findimpossible(shrink);
        if(impossiblelist.equals("nofile") == false) {
            // read impoosible list from file, 1 is impossible, 0 is possible.
            int[][] impossible = vadata.addphyimpossible(dir0 + impossiblelist + ".csv");
            model.findimpossible(impossible);
        }




        if (type.equals("PX")) {
            model.cov_px.initial_hotstart(model, 0.1);
            model.setanneal(anneal);
            model.fit_PX_model(seed, true, is_this_classification_job, integrate, NB, same_pop);
            EvalUtil.save(model, model.cov_px, currentdir, currentfile, "PX", true);
        } else if (type.equals("SSSL")) {

            double[][] alpha = new double[P][P];
            double[][] beta = new double[P][P];
            for (int i = 0; i < P; i++) {
                for (int j = 0; j < P; j++) {
                    alpha[i][j] = 1;
                }
            }
            for (int i = 0; i < P; i++) {
                for (int j = 0; j < P; j++) {
                    beta[i][j] = alpha[i][j] * (1 - 1.0 * prob) / (1.0 * prob);
                }
            }
            model.cov_sssl.sethyper(G, v0, v1, lambda, alpha, beta);
            model.setanneal(anneal);
            model.cov_sssl.initial_hotstart(model, 0.1);
            model.fit_SSSL_model(seed, update_sparsity, true, is_this_classification_job, integrate, NB, same_pop,
                    currentdir, currentfile);
            if(model.savefull){
                EvalUtil.save_full(model, model.cov_sssl, currentdir, currentfile, "SSSL", true);
            }
            EvalUtil.save(model, model.cov_sssl, currentdir, currentfile, "SSSL", true);

        }
        System.out.println(Arrays.toString(args));

    }


    /**
     * add data from file Without Header but with ID
     *
     * @param filename_mar marginal infor file name
     * @param filename     file name to read in data, comma separated, no header
     *                     cause starts from 1
     * @param P            dimension of data
     * @throws IOException
     */
    public void addData(String filename_mar, String filename, int P, int seed, int subgroup) throws IOException {
        addData(filename_mar, filename, P, Integer.MAX_VALUE, seed, subgroup);
    }

    public void addinitialR( String filename, boolean hotstart) throws IOException {

        if(hotstart) {
            BufferedReader br0 = new BufferedReader(new FileReader(filename));
            LinkedList<String[]> rows_mar = new LinkedList<String[]>();
            String line;
            while ((line = br0.readLine()) != null) {
                rows_mar.addLast(line.split(","));
            }
            br0.close();
            this.Rinit = new double[this.P][this.P];
            for (int i = 0; i < this.P; i++) {
                for (int j = 0; j < this.P; j++) {
                    this.Rinit[i][j] = Double.parseDouble(rows_mar.get(i)[j]);
                }
            }
        }else{
            this.Rinit = new double[this.P][this.P];
            for (int i = 0; i < this.P; i++) {
                    this.Rinit[i][i] = 1;
            }
        }

    }

    public static double[] readPriorCSMF( String filename) throws IOException {
        BufferedReader br0 = new BufferedReader(new FileReader(filename));
        LinkedList<String[]> rows_mar = new LinkedList<String[]>();
        String line;
        while ((line = br0.readLine()) != null) {
            rows_mar.addLast(line.split(","));
        }
        br0.close();
        double[] csmf = new double[rows_mar.size()];

        for (int i = 0; i < rows_mar.size(); i++) {
            csmf[i] = Double.parseDouble(rows_mar.get(i)[0]);
        }
       return(csmf);
    }


    public void addData(String filename_mar, String filename, int P, int max, int seed, int subgroup) throws
            IOException {


        BufferedReader br0 = new BufferedReader(new FileReader(filename_mar));
        LinkedList<String[]> rows_mar = new LinkedList<String[]>();
        String line;
        while ((line = br0.readLine()) != null) {
            rows_mar.addLast(line.split(","));
        }
        br0.close();
        if(P == 0) P = rows_mar.get(0).length;
        if(subgroup == Integer.MAX_VALUE){
            this.P = P;
            this.G = rows_mar.size();
            this.delta = new double[this.G][this.P];
            for(int i = 0; i < this.G; i++){
                for(int j = 0; j < this.P; j++){
                    this.delta[i][j] = -Double.parseDouble(rows_mar.get(i)[j]);
                }
            }
            if(max > 0){
                BufferedReader br = new BufferedReader(new FileReader(filename));
                LinkedList<String[]> rows = new LinkedList<String[]>();
                while ((line = br.readLine()) != null) {
                    rows.addLast(line.split(","));
                }
                br.close();

                this.Ntrain = Math.min(max, rows.size());
                this.data = new double[this.Ntrain][this.P];
                this.cause = new int[Ntrain];

                List<Integer> shuffle = new ArrayList<Integer>();
                for(int i = 0; i < rows.size(); i++) shuffle.add(i);
                Random rand = new Random(seed);
                Collections.shuffle(shuffle, rand);

                for(int i = 0 ; i < this.Ntrain; i++){
                    if(i >= max){break;}
                    int kk = shuffle.get(i);
                    this.cause[i] = Integer.parseInt(rows.get(kk)[0]) - 1;
                    for(int j = 0; j < this.P; j++){
                        if(rows.get(kk)[j + 1].equals("Y")){
                            this.data[i][j] = 1;
                        }else if(rows.get(kk)[j + 1].equals(".")){
                            this.data[i][j] = -Double.MAX_VALUE;
                        }else{
                            this.data[i][j] = 0;
                        }
                    }
                    if(i % 100 == 0) System.out.print(".");
                }
                System.out.println("Number of rows read from data: " + Ntrain + "\n");
            }
        }else{
            this.P = P;
            this.G = 1;
            this.delta = new double[this.G][this.P];
            for(int j = 0; j < this.P; j++){
                this.delta[0][j] = -Double.parseDouble(rows_mar.get(subgroup-1)[j]);
            }
            BufferedReader br = new BufferedReader(new FileReader(filename));
            LinkedList<String[]> rows = new LinkedList<String[]>();
            int count = 0;
            int index = 0;
            while ((line = br.readLine()) != null) {
                rows.addLast(line.split(","));
                if(Integer.parseInt(rows.get(index)[0]) == subgroup){
                    count++;
                    index++;
                }else{
                    index++;
                }
            }
            br.close();

            this.Ntrain = Math.min(max, count);
            this.data = new double[this.Ntrain][this.P];
            this.cause = new int[Ntrain];

            List<Integer> shuffle = new ArrayList<Integer>();
            for(int i = 0; i < rows.size(); i++) shuffle.add(i);
            Random rand = new Random(seed);
            Collections.shuffle(shuffle, rand);

            index=0;
            for(int i = 0 ; i < rows.size(); i++){
                if(Integer.parseInt(rows.get(i)[0]) != subgroup){continue;}
                this.cause[index] = 0;
                for(int j = 0; j < this.P; j++){
                    if(rows.get(i)[j + 1].equals("Y")){
                        this.data[index][j] = 1;
                    }else if(rows.get(i)[j + 1].equals(".")){
                        this.data[index][j] = -Double.MAX_VALUE;
                    }else{
                        this.data[index][j] = 0;
                    }
                }
                index++;
                if(i % 100 == 0) System.out.print(".");
            }
            System.out.println("Number of rows read from data: " + Ntrain + "\n");
        }

    }

    public void addTestData(String filename, int P) throws IOException {
        addTestData(filename, P, Integer.MAX_VALUE);
    }
    public void addTestData( String filename, int P, int max) throws IOException {
        String line;
        BufferedReader br = new BufferedReader(new FileReader(filename));
        LinkedList<String[]> rows = new LinkedList<String[]>();
        while ((line = br.readLine()) != null) {
            if(rows.size() >= max){break;}
            rows.addLast(line.split(","));
        }
        br.close();
        this.N_test = rows.size();
        this.data_test = new double[this.N_test][this.P];
        this.cause_test = new int[N_test];

        for(int i = 0 ; i < this.N_test; i++){
            if(i >= max){break;}
            this.cause_test[i] = Integer.parseInt(rows.get(i)[0]) - 1;
            for(int j = 0; j < this.P; j++){
                if(rows.get(i)[j + 1].equals("Y")){
                    this.data_test[i][j] = 1;
                }else if(rows.get(i)[j + 1].equals(".")){
                    this.data_test[i][j] = -Double.MAX_VALUE;
                }else{
                    this.data_test[i][j] = 0;
                }
            }
            if(i % 100 == 0) System.out.print(".");
        }
        System.out.println("Number of rows read from test data: " + N_test + "\n");
    }
    public int[][] addphyimpossible( String filename) throws IOException {
        int[][] graph = new int[N_test][G];
        double sum = 0;
        BufferedReader br0 = new BufferedReader(new FileReader(filename));
        LinkedList<String[]> rows_mar = new LinkedList<String[]>();
        String line;
        while ((line = br0.readLine()) != null) {
            rows_mar.addLast(line.split(","));
        }
        br0.close();
        for (int i = 0; i < N_test; i++) {
            for (int j = 0; j < G; j++) {
                graph[i][j] = Integer.parseInt(rows_mar.get(i)[j]);
                sum += graph[i][j];
            }
        }
        System.out.println("Total number of known causes removed: " + sum);
        return(graph);
    }
    public double[][] addgraph( String filename, int P, boolean notexist) throws IOException {
        double[][] graph = new double[P][P];
        if(notexist) return(graph);
        double sum = 0;
        BufferedReader br0 = new BufferedReader(new FileReader(filename));
        LinkedList<String[]> rows_mar = new LinkedList<String[]>();
        String line;
        while ((line = br0.readLine()) != null) {
            rows_mar.addLast(line.split(","));
        }
        br0.close();
        for (int i = 0; i < P; i++) {
            for (int j = 0; j < P; j++) {
                graph[i][j] = Double.parseDouble(rows_mar.get(i)[j]);
                sum += graph[i][j];
            }
        }
        System.out.println("Total number of known edges: " + sum);
        return(graph);
    }
//
//    public double[] getDelta(){
//        int[] index = new int[this.N];
//        for(int i = 0; i < this.N; i++) index[i]=i;
//        return(this.getDelta(index));
//    }
//
//    public double[] getDelta_train(){
//        NormalDistribution rngN = new NormalDistribution(0, 1);
//        double[] delta = new double[this.train[0].length];
//        double[] size = new double[this.train[0].length];
//        for(int j = 0; j < this.train[0].length; j++){
//            for(int i = 0; i < this.train.length; i++){
//                delta[j] += (this.train[i][j] == -Double.MAX_VALUE) ? 0 : this.train[i][j];
//                size[j] += (this.train[i][j] == -Double.MAX_VALUE) ? 0 : 1;
//            }
//            delta[j] /= size[j];
//
//            // process the mean of the empirical marginals to the scale we want
//
//            if(type[j] == 0){
//                // todo: continuous case
//            }else if(type[j] == 1){
//                // binary case
//                delta[j] = rngN.inverseCumulativeProbability(1 - delta[j]);
//            }else if(type[j] == 2){
//                //todo: categoricla case
//            }
//        }
//
//        return(delta);
//    }
//
//    public double[][] getDelta_train_by_cause(){
//        NormalDistribution rngN = new NormalDistribution(0, 1);
//        double[][] delta = new double[this.Gnew][this.train[0].length];
//        double[][] size = new double[this.Gnew][this.train[0].length];
//        for(int j = 0; j < this.train[0].length; j++){
//            for(int i = 0; i < this.train.length; i++){
//                delta[this.cause_i_train[i]][j] += (this.train[i][j] == -Double.MAX_VALUE) ? 0 : this.train[i][j];
//                size[this.cause_i_train[i]][j] += (this.train[i][j] == -Double.MAX_VALUE) ? 0 : 1;
//            }
//            for(int g = 0; g < this.Gnew; g++){
//                delta[g][j] /= size[g][j];
//            }
//
//            // process the mean of the empirical marginals to the scale we want
//
//            if(type[j] == 0){
//                // todo: continuous case
//            }else if(type[j] == 1){
//                // binary case
//                for(int g = 0; g < this.Gnew; g++){
//                    delta[g][j] = rngN.inverseCumulativeProbability(1 - delta[g][j]);
//                }
//            }else if(type[j] == 2){
//                //todo: categoricla case
//            }
//        }
//
//        return(delta);
//    }
//
//    public double[] getDelta(int[] index){
//        NormalDistribution rngN = new NormalDistribution(0, 1);
//        double[] delta = new double[this.P];
//        double[] size = new double[this.P];
//        for(int j = 0; j < this.P; j++){
//            for(int i : index){
//                delta[j] += (this.data[i][j] == -Double.MAX_VALUE) ? 0 : this.data[i][j];
//                size[j] += (this.data[i][j] == -Double.MAX_VALUE) ? 0 : 1;
//            }
//            delta[j] /= size[j];
//
//            // process the mean of the empirical marginals to the scale we want
//
//            if(type[j] == 0){
//                // todo: continuous case
//            }else if(type[j] == 1){
//                // binary case
//                delta[j] = rngN.inverseCumulativeProbability(1 - delta[j]);
//            }else if(type[j] == 2){
//                //todo: categoricla case
//            }
//        }
//
//        return(delta);
//    }
//
//    public void pick_site(String test_site){
//        int Ntrain = 0, Ntest = 0;
//        int[] count_train = new int[this.P];
//        int[] count_test = new int[this.P];
//        this.isTrain = new boolean[this.P];
//        for(int i = 0; i < this.N; i++){
//            this.isTrain[i] = !this.site[i].equals(test_site);
//            if(this.site[i].equals(test_site)){
//                Ntest++;
//                for(int j=0; j<this.P; j++) count_test[j] += this.data[i][j] == -Double.MAX_VALUE ? 0 : this.data[i][j];
//            }else{
//                Ntrain++;
//                for(int j=0; j<this.P; j++) count_train[j] += this.data[i][j] == -Double.MAX_VALUE ? 0 : this
//                        .data[i][j];
//            }
//        }
//
//        boolean[] remove = new boolean[this.P];
//        int Pnew = 0;
//        for(int i = 0; i < this.P; i++){
//            remove[i] = (count_test[i] == 0) |(count_test[i] == Ntest) |(count_train[i] == 0) |(count_train[i] ==
//                    Ntrain);
//            Pnew += remove[i] ? 0 : 1;
//        }
//        System.out.println("New dimensionality after train/test split: " + Pnew);
//
//        this.test = new double[Ntest][Pnew];
//        this.train = new double[Ntrain][Pnew];
//
//        int counter0=0;
//        int counter1=0;
//        for(int i=0; i < this.N; i++){
//            if(isTrain[i]){
//                int counterP=0;
//                for(int j=0; j < this.P; j++){
//                    if(!remove[j]){
//                        this.train[counter0][counterP] = data[i][j];
//                        counterP++;
//                    }
//                }
//                counter0++;
//            }else{
//                int counterP=0;
//                for(int j=0; j < this.P; j++) {
//                    if(!remove[j]) {
//                        this.test[counter1][counterP] = data[i][j];
//                        counterP++;
//                    }
//                }
//                counter1++;
//            }
//        }
//
//
//        int counter = 0;
//        int counter_train=0;
//        int counter_test=0;
//        this.cause_look_up = new HashMap<>();
//        this.index_look_up = new HashMap<>();
//        this.Gnew = 0;
//        for(int i = 0; i < this.N; i++) {
//            if (isTrain[i]) {
//                if (index_look_up.get(this.cause[i]) == null) {
//                    this.index_look_up.put(this.cause[i], counter);
//                    this.cause_look_up.put(counter, this.cause[i]);
//                    this.Gnew++;
//                    counter++;
//                }
//            }
//        }
//
//        this.cause_i_train = new int[Ntrain];
//        this.cause_i_test = new int[Ntest];
//        counter_train = 0;
//        counter_test = 0;
//        for(int i = 0; i < this.N; i++) {
//            if (isTrain[i]) {
//                this.cause_i_train[counter_train] = this.index_look_up.get(this.cause[i]);
//                counter_train++;
//            }else{
//                this.cause_i_test[counter_test] = this.index_look_up.get(this.cause[i]);
//                counter_test++;
//            }
//        }
//
//
//    }
//
//
//    public static void main(String[] args) throws IOException {
//        ProcessVAdata vadata = new ProcessVAdata();
//        vadata.addData("data/phmrc_clean_with_cause_site_mar.csv", 177, 100);
//        vadata.pick_site("Mexico");
//        double[] delta =  vadata.getDelta_train();
////        for(int i = 0; i < vadata.P; i++){
////            if(delta[i] == Double.POSITIVE_INFINITY) System.out.println(i);
////        }
//        System.out.println(Arrays.toString(delta));
//
//    }
}
