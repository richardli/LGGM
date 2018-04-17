package util;

import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
import sampler.Latent_classifier;
import sampler.Latent_model;

import java.io.IOException;
import java.util.Random;

/**
 * Created by zehangli on 1/25/18.
 */
public class ExploreVA extends ProcessVAdata {

    public static void main(String[] args) throws IOException {

        ProcessVAdata vadata = new ProcessVAdata();
        int counter = 0;
        String prior = args[counter];
        counter++;
        String train = args[counter];
        counter++;
        String directory = args[counter];
        counter++;
        int Nitr = Integer.parseInt(args[counter]);
        counter++;
        String name = args[counter];
        counter++;
        String type = args[counter];
        counter++;
        double sd0 = Double.parseDouble(args[counter]);
        counter++;
        boolean NB = Boolean.parseBoolean(args[counter]);
        counter++;
        boolean integrate = false;
        if (args.length > counter) integrate = Boolean.parseBoolean(args[counter]);
        counter++;
        int PP = 0;
        if (args.length > counter) PP = Integer.parseInt(args[counter]);
        counter++;
        int seed = 1;
        if (args.length > counter) seed = Integer.parseInt(args[counter]);
        counter++;
        int Ntrain = Integer.MAX_VALUE;
        if (args.length > counter) Ntrain = Integer.parseInt(args[counter]);
        counter++;
        boolean update_with_test = true;
        if (args.length > counter) update_with_test = Boolean.parseBoolean(args[counter]);
        counter++;
        boolean adaptive = false;
        if (args.length > counter) adaptive = Boolean.parseBoolean(args[counter]);
        counter++;
        double v0 = 0.001;
        double v1 = 10;
        double lambda = 1;
        double prob = 1/(PP+0.0);
        int subgroup = Integer.MAX_VALUE;
        if (args.length > counter) v0 = Double.parseDouble(args[counter]);
        counter++;
        if (args.length > counter) v1 = Double.parseDouble(args[counter]);
        counter++;
        if (args.length > counter) lambda = Double.parseDouble(args[counter]);
        counter++;
        if (args.length > counter) prob = Double.parseDouble(args[counter]);
        counter++;
        if (args.length > counter) subgroup = Integer.parseInt(args[counter]);
        counter++;
        boolean is_this_classification_job = false;
        boolean same_pop = false;
        boolean hotstart = false;
        double a_sd0 = 0.001;
        double b_sd0 = 0.001;

        String dir0 = "../data/";
        if(directory.equals("/Users/zehangli/")) dir0 = "/Users/zehangli/Bitbucket-repos/LatentGaussian/data/";

        vadata.addData(dir0 + prior + "_delta.csv", dir0 + train + ".csv",PP, Ntrain, seed, subgroup);
        vadata.addinitialR("to-remove-this-initialization-place-holder", hotstart);


        // sd0 = 0.5;
        int N = vadata.Ntrain;
        int P = vadata.P;
        int G = vadata.G;
        double power = 0;
        boolean update_sparsity = false;


        String expriment_name = name;
        String currentdir = directory + expriment_name + "/";
        String currentfile = expriment_name;

        Latent_model model = new Latent_model(Nitr, N, 0, P, G, type);
        model.data.init_adaptive(sd0, a_sd0, b_sd0, adaptive, power);
        model.data.N = N;
        model.data.G = vadata.G;
        model.data.ID = new String[model.data.N];
        model.data.data = new double[model.data.N][vadata.P];
        model.data.latent = new double[model.data.N][vadata.P];
        model.data.expanded = new double[model.data.N][vadata.P];
        model.data.Delta = new double[G][vadata.P];
        model.data.Delta_expanded = new double[G][vadata.P];

        // need to update this when cont data are included
        model.data.type = new int[vadata.P];
        for (int i = 0; i < vadata.P; i++) {
            model.data.type[i] = 1;
        }
        Random rand = new Random(seed);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++) {
                model.data.data[i][j] = vadata.data[i][j];
            }
            model.data.membership[i] = vadata.cause[i];
            model.data.ID[i] = "id" + i;
        }
        for (int j = 0; j < P; j++) {
            for (int g = 0; g < G; g++) {
                model.data.mu[g][j] = vadata.delta[g][j] ;// * (sd0 * sd0 + 1);
                model.data.Delta[g][j] = vadata.delta[g][j] ;// * (sd0 * sd0 + 1);
            }
            model.data.type[j] = 1;
        }

        DoubleMersenneTwister rngEngine = new DoubleMersenneTwister(seed);
        Gamma rngG = new Gamma(1.0, 1.0, rngEngine);
        model.data.update_groupcounts(model.update_with_test);


        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++) {
                model.data.latent[i][j] = model.data.data[i][j] == 0 ? -1 : model.data.data[i][j];
                model.data.latent[i][j] = model.data.data[i][j] == -Double.MAX_VALUE ? 0 : model.data.data[i][j];
//                if(model.data.latent[i][j] * vadata.delta[model.data.membership[i]][j] < 0){
//                    model.data.latent[i][j] = 0;
//                }
                model.data.expanded[i][j] = model.data.latent[i][j];
            }
        }

        if (type.equals("PX")) {
            model.cov_px.initial_hotstart(model, 0.1);
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
            model.cov_sssl.initial_hotstart(model, 0.1);
            model.fit_SSSL_model(seed, update_sparsity, true, is_this_classification_job, integrate, NB, same_pop,
                    currentdir, currentfile);
            EvalUtil.save(model, model.cov_sssl, currentdir, currentfile, "SSSL", true);
        }

    }


}
