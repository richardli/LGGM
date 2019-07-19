package sampler;

import cern.jet.random.tdouble.Exponential;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.MersenneTwister;
import util.EvalUtil;
import util.Simulator;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by zehangli on 5/20/16.
 */
public class Latent_model {

    public int Nitr;
    public int burnin;
    public int thin;
    public int Nout;
    public int N;
    public int N_test;
    public int P;
    public int G;
    public Mix_data data;
    String covType;
    public PX_COV_model cov_px;
    public SSSL_COV_model cov_sssl;
    public Glasso_COV_model cov_glasso;
    public double[] true_prob;
    public double[] post_prob;
    public double[] post_prob_theta_transform;
    public double[] prior_mean_theta_transform;
    public double var0;

    public double[][][] corr_draw;
    public double[][][] prec_draw;
    public double[][] sd_draw;
    public double[][][] mean_draw;
    public int[] inclusion_draw;

    // field for separating continuous dimensions
    public int P_binary;
    public int P_cont;
    public int[] binary_indices;
    public int[] cont_indices;
    public boolean Dirichlet;

    // field used only in classifier method
    public boolean[] test_case;
    public double[][][] post_prob_draw;
    public double[][] post_prob_draw_mean;
    public double[][] post_prob_pop;
    public boolean update_with_test;
    boolean anneal;
    double[][] penalty_mat;
    public boolean savefull;

    public void init_output(){
        this.corr_draw = new double[this.Nout][this.P][this.P];
        this.prec_draw = new double[this.Nout][this.P][this.P];
        this.sd_draw = new double[this.Nout][this.P];
        this.mean_draw = new double[this.G][this.Nout][this.P];
        this.inclusion_draw = new int[this.Nout];
    }

    public Latent_model(int Nitr, int burnin, int thin, int N, int P, int G, String covType){
        this.Nitr = Nitr;
        this.burnin = burnin;
        this.thin = thin;
        this.P = P;
        this.N = N;
        this.N_test = 0;
        this.Nout = ((int) ((this.Nitr-this.burnin) / (this.thin + 0.0)));
        // only one group default
        this.G = G;
        this.data = new Mix_data(N, this.N_test,  P, G);
        this.anneal = false;

        this.init_cov(covType);
        this.init_output();
        this.savefull=false;
    }
    public Latent_model(int Nitr, int burnin, int thin, int N, int N_test, int P, int G, String covType){
        this.Nitr = Nitr;
        this.burnin = burnin;
        this.thin = thin;
        this.P = P;
        this.N = N;
        this.N_test = N_test;
        this.Nout = ((int) ((this.Nitr-this.burnin) / (this.thin + 0.0)));
        this.penalty_mat = new double[this.P][this.P];

        // only one group default
        this.G = G;
        this.data = new Mix_data(N, N_test, P, G);
        this.anneal = false;

        this.init_cov(covType);
        this.init_output();
        this.savefull=false;
    }

    public Latent_model(int Nitr, int N,  int N_test, int P,  int G, String covType){
        this.Nitr = Nitr;
        this.burnin = (int) (Nitr / 2.0);
        this.thin = Nitr > 5000 ? 10 : 1;
        this.P = P;
        this.N = N;
        this.N_test = N_test;
        this.G = G;
        this.Nout = ((int) ((this.Nitr-this.burnin) / (this.thin + 0.0)));
        this.penalty_mat = new double[this.P][this.P];

        this.data = new Mix_data(N, N_test, P, G);
        this.anneal = false;

        this.init_cov(covType);
        this.init_output();
        this.savefull=false;
    }

    // default case with no grouping
    public Latent_model(int Nitr, int N, int P, String covType){
        this.Nitr = Nitr;
        this.burnin = (int) (Nitr / 2.0);
        this.thin = 1;
        this.P = P;
        this.N = N;
        this.Nout = ((int) ((this.Nitr-this.burnin) / (this.thin + 0.0)));
        this.penalty_mat = new double[this.P][this.P];

        this.G = 1;
        this.data = new Mix_data(N, N_test, P, G);
        this.anneal = false;

        this.init_cov(covType);
        this.init_output();
        this.savefull=false;
    }

    public void init_cov(String covType){
        this.covType = covType;
        if(covType.equals("PX")){
            this.cov_px = new PX_COV_model(covType, P);
        }else if(covType.equals("SSSL")){
            this.cov_sssl = new SSSL_COV_model(covType, P);
        }else if(covType.equals("GLASSO")){
            this.cov_glasso = new Glasso_COV_model(covType, P);
        }
    }

    public void update_variable_type(COV_model cov_model){
        this.P_binary = 0;
        ArrayList<Integer> binarys = new ArrayList<>();
        ArrayList<Integer> conts = new ArrayList<>();
        for(int j = 0; j < this.P; j++){
            if(this.data.type[j] == 1){
                this.P_binary ++;
                binarys.add(j);
            }else{
                this.P_cont++;
                conts.add(j);
            }
        }

        this.binary_indices = new int[binarys.size()];
        this.cont_indices = new int[conts.size()];
        for(int i = 0; i < binarys.size(); i++) this.binary_indices[i] = binarys.get(i);
        for(int i = 0; i < conts.size(); i++) this.cont_indices[i] = conts.get(i);

        if(this.cont_indices.length > 0){
            double[] means = new double[cont_indices.length];
            for(int j : this.cont_indices){
                cov_model.tau[j] = 0;
                double count = 0;
                for(int i=0; i < this.N; i++){
                    count += this.data.data[i][j] == -Double.MAX_VALUE ? 0:1;
                    means[j] += this.data.data[i][j] == -Double.MAX_VALUE ? 0:this.data.data[i][j];
                }
                for(int i=0; i < this.N; i++){
                    cov_model.tau[j] += this.data.data[i][j] == -Double.MAX_VALUE ? 0:Math
                            .pow(this.data.data[i][j] - means[j]/count, 2);
                }
                cov_model.tau[j] = Math.sqrt(cov_model.tau[j] / count);
            }
        }
    }

    public void read_true_cov(String covType, String covfile, String precfile) throws IOException {
        if(covType.equals("PX")){
            this.cov_px.readSigma(covfile);
            this.cov_px.readOmega(precfile);
        }else if(covType.equals("SSSL")){
            this.cov_sssl.readSigma(covfile);
            this.cov_sssl.readOmega(precfile);
        }else if(covType.equals("GLASSO")) {
            this.cov_glasso.readSigma(covfile);
            this.cov_glasso.readOmega(precfile);
        }
    }



    // read file from file name, marginal file name
    public void readData(String filename, String marfilename) throws IOException {
        this.data.addData(filename, marfilename, this.P);
    }


    public void fit_PX_model(int seed, boolean verbose, boolean update_group, boolean integrate,
                             boolean NB, boolean same_pop){

        DoubleMersenneTwister rngEngine = new DoubleMersenneTwister(seed);
        MersenneTwister rngEngine2 = new org.apache.commons.math3.random
                .MersenneTwister(seed+1);
        NormalDistribution rngN = new NormalDistribution(rngEngine2, 0, 1);
        Gamma rngG = new Gamma(1.0, 1.0, rngEngine);
        Exponential rngE = new Exponential(1, rngEngine);
        Random rand = new Random(seed+2);
//        this.cov_px.sethyper(sigma_px);

        this.N = this.data.N;
        // initial latent mean to prior
        this.data.initial_Delta(this.data.mu);
        // initial latent variables
        this.data.initial_Z(rand, rngN, rngE,
                this.data.Delta, this.data.d, true);
        this.update_variable_type(this.cov_px);


        // get initial norms
        double[] metrics0 = EvalUtil.getnorm(this.cov_px.corr,
                this.cov_px.cov_true);
        if(verbose){
            System.out.printf("Initial:\n" +
                    "F-norm: %.4f, Infty-norm: %.4f, M-norm %.4f\n",
                    metrics0[0], metrics0[1], metrics0[2]);
        }

        /** Start of MCMC loop **/
        for(int itr = 0; itr < Nitr; itr ++) {
            System.out.println("Iteration " + itr);
            double[][] testprob = new double[this.N_test][this.G];
            if((update_group & itr > burnin) | this.update_with_test){
                double[][] tmp = this.cov_px.resample(this, this.data, rand, rngEngine2, rngN, rngE, rngG,
                        true, integrate, NB, same_pop, itr, Nitr);
                for(int i = 0; i < this.N_test; i++){
                    for(int j = 0; j < this.G; j++){
                        testprob[i][j] = tmp[i][j];
                    }
                }
            }else{
                this.cov_px.resample(this.data, rand, rngEngine2, rngN, rngE, rngG, true, this.update_with_test, this
                        .binary_indices, this.cont_indices);
            }

            if(itr > burnin &
                    (itr - this.burnin) % (this.thin + 0.0) == 0){
                if(update_group){
                    this.save_output(itr, verbose, testprob);
                }else {
                    this.save_output(itr, verbose);
                }
            }else{
                if(update_group) this.save_output_test_prob_only(itr, testprob);
            }
        }
    }


    public void fit_SSSL_model(int seed, boolean update_sparsity, boolean verbose, boolean
            update_group, boolean integrate, boolean NB, boolean same_pop, String currentdir, String currentfile) throws IOException {
        fit_SSSL_model(seed, this.cov_sssl.v0, this.cov_sssl.v1, this.cov_sssl.lambda,
                this.cov_sssl.alpha, this.cov_sssl.beta, update_sparsity, verbose, update_group, integrate, NB,
                same_pop, currentdir, currentfile);
    }

    public void fit_SSSL_model(int seed, double v0, double v1, double lambda, double[][] alpha, double[][] beta, boolean update_sparsity, boolean verbose, boolean update_group, boolean integrate, boolean NB, boolean same_pop, String currentdir, String currentfile) throws IOException {
        DoubleMersenneTwister rngEngine = new DoubleMersenneTwister(seed);
        MersenneTwister rngEngine2 = new org.apache.commons.math3.random
                .MersenneTwister(seed + 1);
        NormalDistribution rngN = new NormalDistribution(rngEngine2, 0, 1);
        Gamma rngG = new Gamma(1.0, 1.0, rngEngine);
        Exponential rngE = new Exponential(1, rngEngine);
        Random rand = new Random(seed + 2);
        this.cov_sssl.sethyper(this.G, v0, v1, lambda, alpha, beta);
        this.N = this.data.N;

        RealMatrix x = new Array2DRowRealMatrix(this.data.data);
        for (int i = 0; i < this.N; i++) {
            for (int j = 0; j < this.P; j++) {
                if (x.getEntry(i, j) == 0) x.setEntry(i, j, -0.5);
                if (x.getEntry(i, j) == 1) x.setEntry(i, j, 0.5);
                if (x.getEntry(i, j) < -1e100) x.setEntry(i, j, 0);
            }
        }
        RealMatrix covini = x.preMultiply(x.transpose()).scalarMultiply(1 / (this.data.N + 0.0));
        this.cov_sssl.setcov(covini, 0.5);
        this.update_variable_type(this.cov_sssl);

        // initial latent mean to prior
        this.data.initial_Delta(this.data.mu);
        // initial latent variables
        this.data.initial_Z(rand, rngN, rngE, this.data.Delta, this.data.d, true);
        // get initial norms
        double[] metrics0 = EvalUtil.getnorm(this.cov_sssl.corr,
                this.cov_sssl.cov_true);
        int[] metrics1 = EvalUtil.getclassification(this.cov_sssl.inclusion, this.cov_sssl.prec_true);
        if (verbose) {
            System.out.printf("Initial:\n" +
                            "F-norm: %.4f, Infty-norm: %.4f, M-norm %.4f\n",
                    metrics0[0], metrics0[1], metrics0[2]);
            System.out.printf("Initial:\n" +
                            "TP: %d, FP: %d, TN: %d, FN: %d\n",
                    metrics1[0], metrics1[1], metrics1[2], metrics1[3]);
        }

        /** Start of MCMC loop **/
        for (int itr = 0; itr < Nitr; itr++) {
            System.out.println("Iteration " + itr);
            double[][] testprob = new double[this.N_test][this.G];

            if ((update_group & itr > burnin) | this.update_with_test) {
                double[][] tmp = this.cov_sssl.resample_SSSL(this, this.data,
                        rand, rngEngine2, rngN, rngE, rngG, update_sparsity, verbose,
                        integrate, NB, same_pop, itr, Nitr);
                for (int i = 0; i < this.N_test; i++) {
                    for (int j = 0; j < this.G; j++) {
                        testprob[i][j] = tmp[i][j];
                    }
                }
            } else {
                this.cov_sssl.resample_SSSL(this.data,
                        rand, rngEngine2, rngN, rngE, rngG, update_sparsity, verbose, this.update_with_test, this
                                .binary_indices, this.cont_indices, this.penalty_mat);
            }

            if (itr > burnin &
                    (itr - this.burnin) % (this.thin + 0.0) == 0) {
                if (update_group) {
                    this.save_output(itr, verbose, testprob);
                } else {
                    this.save_output(itr, verbose);
                }
            }else{
                if(update_group) this.save_output_test_prob_only(itr, testprob);
            }
        }

        if (this.data.G > 1) {
            if(savefull){
                EvalUtil.save_full(this, this.cov_sssl, currentdir, currentfile + "_s1", covType, true);
            }
            EvalUtil.save(this, this.cov_sssl, currentdir, currentfile + "_s1", covType, true);
        }
    }

    // todo: Glasso estimator
    public void fit_Glasso_model(int seed, boolean update_group){

    }

    // not used
    public double[][] update_group(Random rand, Gamma rngG, NormalDistribution rngN, Exponential rngE, boolean
            integrate, boolean
            NB, boolean
            same_pop, double temp){
        return(new double[1][1]);
    }
    public double[][] update_group( Random rand, Gamma rngG, NormalDistribution rngN, Exponential rngE, boolean integrate, boolean NB, boolean
            same_pop, double temp, boolean updateprob){
        return(new double[1][1]);
    }
    public double[][] update_group_marginal( Random rand, Gamma rngG, NormalDistribution rngN, Exponential rngE,
                                             boolean integrate, boolean
                                                     NB, boolean
                                                     same_pop, double temp, boolean updateprob) {
        return(new double[1][1]);
    }

    public void save_output_test_prob_only(int itr, double[][] test_prob){
        for (int g = 0; g < G; g++) {
            this.post_prob_pop[g][itr] = this.post_prob[g];
        }
    }

    public void save_output(int itr, boolean verbose, double[][] test_prob) {
        save_output(itr, verbose);
            int current_count = (int) ((itr - this.burnin) / (this.thin + 0.0));
            int counter = 0;
            for (int i = 0; i < this.N; i++) {
                if (this.test_case[i]) {
                    for (int g = 0; g < G; g++) {
                        this.post_prob_draw[g][current_count][i] = test_prob[counter][g];
                        this.post_prob_draw_mean[g][i] += test_prob[counter][g];
                    }
                    counter++;
                }
            }
            for (int g = 0; g < G; g++) {
                this.post_prob_pop[g][itr] = this.post_prob[g];
            }
        }

    public void save_output(int itr, boolean verbose){
            int current_count = (int) ((itr - this.burnin) / (this.thin + 0.0));

            double[] metrics2;
            if(this.covType.equals("PX")){
                EvalUtil.save_draw_px(this, current_count, this.G);
                this.cov_px.updateCorrAve();
                this.cov_px.updateInvCorrAve();
                metrics2 = EvalUtil.getnorm(this.cov_px.corr_ave,
                        this.cov_px.cov_true);
            }else if(this.covType.equals("SSSL")){
                EvalUtil.save_draw_sssl(this, current_count, this.G);
                this.cov_sssl.updateCorrAve();
                this.cov_sssl.updateInvCorrAve();
                this.cov_sssl.updateInclusionAve(this.cov_sssl.inclusion);
                metrics2 = EvalUtil.getnorm(this.cov_sssl.corr_ave,
                        this.cov_sssl.cov_true);
                // AUC = EvalUtil.getAUC(this.cov_sssl.prec_true, , this.cov_sssl.inclusion);
            }else if(this.covType.equals("GLASSO")){
                EvalUtil.save_draw_glasso(this, current_count, this.G);
                this.cov_glasso.updateCorrAve();
                this.cov_glasso.updateInvCorrAve();
                this.cov_glasso.updateInclusionAve(this.cov_sssl.inclusion);
                metrics2 = EvalUtil.getnorm(this.cov_glasso.corr_ave,
                        this.cov_glasso.cov_true);
                // AUC = EvalUtil.getAUC(this.cov_glasso.prec_true, , this.cov_glasso.inclusion);
            }else{
                metrics2 = new double[3];
            }

            if(verbose){
                System.out.println("Average ---------------------------------");
                System.out.printf("F-norm: %.4f, Infty-norm: %.4f, M-norm %.4f   ",
                        metrics2[0], metrics2[1], metrics2[2]);
            }
    }


    /**
     * New main method for simulation
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {

        String directory = "/Users/zehangli/";
        String pre = "2test0313";
        int N = 200;
        int P = 50;
        double miss = 0.2;
        int Nrep = 10;
        String covType = "SSSL";
        String covTypeSim = "Random";
        int Nitr = 1000;
        double sd0 = 1;
        int seed = 123;
        boolean informative = false;
        boolean misspecified = true;
        boolean verbose = true;
        boolean transform = true;
        int nContinuous =10;
        int rep0 = 0;
        boolean save_full = true;
        int burnin = 0;
        int thin = 1;
        boolean random_start = false;
        boolean fixed_sim_seed = false;


        if(args.length > 0){
            int counter = 0;
             N = Integer.parseInt(args[counter]); counter++;
             P = Integer.parseInt(args[counter]); counter++;
             miss = Double.parseDouble(args[counter]); counter++;
             nContinuous = Integer.parseInt(args[counter]); counter++;
             Nrep = Integer.parseInt(args[counter]); counter++;
             covType = args[counter]; counter++;
             covTypeSim = args[counter]; counter++;
             Nitr = Integer.parseInt(args[counter]); counter++;
             sd0 = Double.parseDouble(args[counter]); counter++;
             seed = Integer.parseInt(args[counter]); counter++;
             informative = Boolean.parseBoolean(args[counter]); counter++;
             misspecified = Boolean.parseBoolean(args[counter]); counter++;
             verbose = Boolean.parseBoolean(args[counter]); counter++;
             transform = Boolean.parseBoolean(args[counter]); counter++;
             directory = args[counter]; counter++;
             pre = args[counter]; counter++;
            // decide if fix a certain rep number
             rep0 = 0;
            if(args.length > counter){
                rep0 = Integer.parseInt(args[counter]);
                counter ++;
            }
            if(rep0 == 0){
                Nrep = 1;
            }
            save_full = Boolean.parseBoolean(args[counter]); counter++;
            if(args.length > counter) {
                burnin = Integer.parseInt(args[counter]); counter++;
            }else{
                burnin = (int) (Nitr / 2.0);
            }
            if(args.length > counter) {
                thin = Integer.parseInt(args[counter]); counter++;
            }else{
                thin = Nitr > 5000 ? 10 : 1;
            }
            if(args.length > counter) {
                random_start = Boolean.parseBoolean(args[counter]); counter++;
            }else{
                random_start = false;
            }
            if(args.length > counter) {
                fixed_sim_seed = Boolean.parseBoolean(args[counter]); counter++;
            }
        }




        boolean update_sparsity = false;
        String expriment_name = pre + covType + "N" + N + "P" + P + "Miss" + miss;

        double c = 0.2;
        double multiplier = 2;
        int G = 1;

        double power = 0;
        boolean adaptive = false;
        double a_sd0 = 0.0001;
        double b_sd0 = 0.0001;
        int seed0 = seed;

        for(int rep = (1 + rep0); rep <= (Nrep + rep0); rep ++) {
            seed = seed + rep * 12345;

            String currentdir = directory + expriment_name + "/";
            String currentfile =  expriment_name + "Rep" + rep;

            Latent_model model = new Latent_model(Nitr,burnin, thin, N,0, P, G, covType);
            model.data.init_adaptive(sd0, a_sd0, b_sd0, adaptive, power);
            Simulator simulator = new Simulator(N, P, G);
            if(fixed_sim_seed){
                Random rand = new Random(seed0);
                MersenneTwister rngEngin = new MersenneTwister(seed0);
                simulator.simCov(P, c, multiplier, seed0, covTypeSim);
                simulator.set_sim(model, rngEngin, rand, covType, miss, informative, misspecified, nContinuous, transform,
                        seed0);
            }else{
                Random rand = new Random(seed);
                MersenneTwister rngEngin = new MersenneTwister(seed);
                simulator.simCov(P, c, multiplier, seed, covTypeSim);
                simulator.set_sim(model, rngEngin, rand, covType, miss, informative, misspecified, nContinuous, transform,
                        seed);
            }

            EvalUtil.savetruth(model, currentdir, currentfile, covType,
                    simulator.prec, simulator.cov, simulator.mean);

            /** Model fitting **/
            if(covType.equals("PX")){
                model.cov_px.initial_hotstart(model, 0.1);
                model.fit_PX_model(seed, verbose, false, false, false, false);
                if(save_full){
                    EvalUtil.save_full(model, model.cov_px, currentdir, currentfile, covType, false);
                }else{
                    EvalUtil.save(model, model.cov_px, currentdir, currentfile, covType, false);
                }
            }else if(covType.equals("SSSL")){
                model.cov_sssl.initial_hotstart(model, 0.1, random_start, seed);
                model.fit_SSSL_model(seed, update_sparsity,  verbose, false, false, false, false, currentdir, currentfile);
                if(save_full){
                    EvalUtil.save_full(model, model.cov_sssl, currentdir, currentfile, covType, false);
                }else{
                    EvalUtil.save(model, model.cov_sssl, currentdir, currentfile, covType, false);
                }


            }else if(covType.equals("GLASSO")){
//                model.fit_Glasso_model(seed, false);
            }
            // save results

        }
    }


}
