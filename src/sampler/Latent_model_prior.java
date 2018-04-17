package sampler;

import cern.jet.random.tdouble.Exponential;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.MersenneTwister;
import util.EvalUtil;
import util.Simulator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

/**
 * Created by zehangli on 3/5/18.
 */
public class Latent_model_prior extends Latent_model{
    ArrayList<Double> spar;

    public Latent_model_prior(int Nitr, int burnin, int thin, int N, int P, int G, String covType) {
        super(Nitr, burnin, thin, N, P, G, covType);
    }

    public void Sample_SSSL_prior(int seed, double v0, double v1, double lambda, double[][] trueprec, double
            sparsity, boolean update_structure){
        DoubleMersenneTwister rngEngine = new DoubleMersenneTwister(seed);
        MersenneTwister rngEngine2 = new org.apache.commons.math3.random
                .MersenneTwister(seed+1);
        NormalDistribution rngN = new NormalDistribution(rngEngine2, 0, 1);
        Gamma rngG = new Gamma(1.0, 1.0, rngEngine);
        Exponential rngE = new Exponential(1, rngEngine);
        Random rand = new Random(seed+2);

        double[][] alpha = new double[P][P];
        double[][] beta = new double[P][P];
        if(sparsity == 0) {
            for (int i = 0; i < P; i++) {
                data.d[i] = 1;
                for (int j = 0; j < P; j++) {
                    alpha[i][j] = 1;
                    if (i != j) sparsity += trueprec[i][j] == 0 ? 0 : 1 / (P * (P - 1) / 2.0);
                }
            }
        }
        for(int i = 0; i < P; i++){
            for(int j = 0; j < P; j++){
                beta[i][j] =  alpha[i][j] * (1 - 1.0 * sparsity) / (1.0 * sparsity);
            }
        }
        this.cov_sssl.sethyper(this.G, v0, v1, lambda, alpha, beta);
        this.N = 0;
        for(int i = 0; i < P; i++){
            data.d[i] = i;
            for(int j = 0; j < P; j++){
                this.cov_sssl.inclusion[i][j] = Math.abs(trueprec[i][j]) < 1E-4 ? 0 : 1;
                this.cov_sssl.prob[i][j] = sparsity;
            }
        }

        /** Start of MCMC loop **/
        this.spar = new ArrayList<>();

        double[][] S = new double[P][P]; // S is a zero matrix
//        for(int i = 0; i < P; i++) S[i][i] = 1;
        RealMatrix S0 = new Array2DRowRealMatrix(S);
        this.penalty_mat = new double[P][P];

        for(int itr = 0; itr < Nitr; itr ++) {
            for(int i = 0; i < P; i++) {
                data.d[i] = Math.sqrt(this.cov_sssl.cov.getEntry(i,i));
            }
            System.out.println("Iteration " + itr);
            // no need this step: since S = 0
//            data.resample_Dfix(rngG, this.cov_sssl.invcorr);

            Collections.shuffle(this.cov_sssl.onetop);
            for (int j : this.cov_sssl.onetop) {
                this.cov_sssl.sample_omega(S0, 0, data, rngN, rngG, j, false, rand, 0);
                if(update_structure){
                    this.cov_sssl.sample_prob(j, false);
                    for (int jj = 0; jj < this.P; jj++) this.cov_sssl.sample_delta(rand, jj, data.d, this.penalty_mat,
                            false);
                }
            }
            if(update_structure){
                int sum = 0;
                for (int jj = 0; jj < this.P; jj++){
                    for(int kk = 0; kk< this.P; kk++){
                        if(kk!=jj) sum += this.cov_sssl.inclusion[jj][kk];
                    }
                }
                System.out.println("Edges: " + sum/2.0 + "  Prob " + sum / (P * (P-1)+0.0));
                this.spar.add(sum / (P * (P-1)+0.0));
            }
            this.cov_sssl.updateCorrFromCov();
            this.cov_sssl.updateInvCorrFromPrecCov();

            if(itr > burnin & (itr - this.burnin) % (this.thin + 0.0) == 0){
                this.save_output(itr, true);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        String directory = "/Users/zehangli/";
        String pre = "complete";
        int N = 0;
        int P = 50;
        double miss = 0.0;
        int Nrep = 1;
        String covType = "SSSL";
        String covTypeSim = "AR1";
        int Nitr = 2000;
        double sd0 = 1;
        int seed = 123;
        boolean informative = false;
        boolean misspecified = false;
        boolean verbose = true;
        boolean transform = true;
        int nContinuous = 0;
        int rep0 = 0;

        boolean update_sparsity = false;
//        String expriment_name = pre + covType + "N" + N + "P" + P + "Miss" + miss;
        Random rand = new Random(seed);
        MersenneTwister rngEngin = new MersenneTwister(seed);
        double c = 0.2;
        double multiplier = 2;
        int G = 1;

        double power = 0;
        boolean adaptive = false;
        double a_sd0 = 2;
        double b_sd0 = 1;

        String expriment_name = "Comp";
        String currentdir = directory + expriment_name + "/";
        double[] v0s = new double[]{0.001, 0.01, 0.1};
        double[] v1s = new double[]{2};
        double[] lambdas = new double[]{1};

        for (double v1 : v1s) {
            for (double lambda : lambdas) {
                String currentfile = expriment_name + "v1" + v1 + "lambda" + lambda;
                Latent_model_prior model = new Latent_model_prior(Nitr, Nitr / 2, 1, 0, P, G, covType);
                model.data.init_adaptive(sd0, a_sd0, b_sd0, adaptive, power);
                Simulator simulator = new Simulator(N, P, G);
                simulator.simCov(P, c, multiplier, seed, covTypeSim);
                simulator.set_sim(model, rngEngin, rand, covType, miss, informative, misspecified, nContinuous, transform,
                        seed);

                EvalUtil.savetruth(model, currentdir, currentfile, covType,
                        simulator.prec, simulator.cov, simulator.mean);
                boolean update_structure = true;
                model.Sample_SSSL_prior(seed, 0.05, v1, lambda, simulator.prec, 1, update_structure);
                EvalUtil.save_full(model, currentdir, currentfile, covType, false);
                EvalUtil.save(model, model.cov_sssl, currentdir, currentfile, covType, false);
            }
        }
//
//        for (double v0 : v0s) {
//            for (double lambda : lambdas) {
//                expriment_name = "AR1";
//                currentdir = directory + expriment_name + "/";
//                double v1 = 1;
//                String currentfile = expriment_name + "v0" + v0 + "lambda" + lambda;
//                Latent_model_prior model = new Latent_model_prior(Nitr, Nitr / 2, 1, 0, P, G, covType);
//                model.data.init_adaptive(sd0, a_sd0, b_sd0, adaptive, power);
//                Simulator simulator = new Simulator(N, P, G);
//                simulator.simCov(P, c, multiplier, seed, covTypeSim);
//                simulator.set_sim(model, rngEngin, rand, covType, miss, informative, misspecified, nContinuous, transform,
//                        seed);
//
//                EvalUtil.savetruth(model, currentdir, currentfile, covType,
//                        simulator.prec, simulator.cov, simulator.mean);
//
//
//                boolean update_structure = false;
//                model.Sample_SSSL_prior(seed, v0, v1, lambda, simulator.prec, 0, update_structure);
//                EvalUtil.save_full(model, currentdir, currentfile, covType, false);
//                EvalUtil.save(model, model.cov_sssl, currentdir, currentfile, covType, false);
//            }
//        }

//        double v0 = 0.01;
//        double v1 = 1;
//        double lambda = 10;
//        if(args.length > 0){
//            System.out.println(Arrays.toString(args));
//            directory = args[0];
//            P = Integer.parseInt(args[1]);
//            v0 = Double.parseDouble(args[2]);
//            v1 = Double.parseDouble(args[3]) * v0;
//            lambda = Double.parseDouble(args[4]);
//            Nitr = Integer.parseInt(args[5]);
//        }
//
//        String expriment_name = "Sparsity";
//        String currentdir = directory + expriment_name + "/";
//        double[] ps = new double[]{0.01, 0.001, 0.0001, 0.00005, 0.00001};
//            for(double p : ps){
//                String currentfile =  expriment_name + v0+ "-" + v1 + "-" + lambda + "-" + p + "-" + P;
//                Latent_model_prior model = new Latent_model_prior(Nitr, Nitr/2 , 1, 0, P, G, covType);
//                model.data.init_adaptive(sd0, a_sd0, b_sd0, adaptive, power);
//                Simulator simulator = new Simulator(N, P, G);
//                simulator.simCov(P, c, multiplier, seed, covTypeSim);
//                simulator.set_sim(model, rngEngin, rand, covType, miss, informative, misspecified, nContinuous, transform,
//                        seed);
//
//                boolean update_structure = true;
//                model.Sample_SSSL_prior(seed, v0, v1, lambda, simulator.prec, p, update_structure);
//                EvalUtil.save(model, model.cov_sssl, currentdir, currentfile, covType, false);
//                EvalUtil.savearray(model.spar, currentdir, currentfile, covType, "sparsity");
//            }
//        }
    }
}

