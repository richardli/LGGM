package util;

import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.random.MersenneTwister;
import sampler.Latent_model;

import java.util.*;

/**
 * Created by zehangli on 10/30/16.
 */
public class Simulator {
    public int N;
    public int P;
    public int G;

    public double[][] X; // raw variables: N x P
    public double[][] Z; // latent Gaussian variables: N x P
    public double[][] prec; // precision matrix: P x P
    public double[][] cov; // covariance matrix: P x P
    public double[][] mean;
    public int[] membership;
    public double[] prob;

    public Simulator(int N, int P, int G){
        this.N = N;
        this.P = P;
        this.G = G;
    }

    public void simCov(int P, double c, double multiplier, int seed, String covTypeSim) {
        Random rand = new Random(seed);
        MersenneTwister rngEngine = new MersenneTwister(seed);
        int[] nEdge = new int[P];
        this.prec = new double[P][P];

        if (covTypeSim.equals("Random")) {
            double[] z1 = new double[P];
            double[] z2 = new double[P];
            for (int i = 0; i < P; i++) {
                z1[i] = rand.nextDouble();
                z2[i] = rand.nextDouble();
            }

            // double c = 0.2;
            // double multiplier =2;
            for (int i = 0; i < P; i++) {
                prec[i][i] = 1;
                for (int j = 0; j < i; j++) {
                    double tmp = Math.sqrt((z1[i] - z1[j]) * (z1[i] - z1[j]) +
                            (z2[i] - z2[j]) * (z2[i] - z2[j]));
                    double prob = Math.pow(2 * Math.PI, -0.5) * Math.exp(-1 * tmp / 2 / c);
                    this.prec[i][j] = rand.nextDouble() < prob ? multiplier : 0;
                    this.prec[j][i] = this.prec[i][j];
                    if (this.prec[i][j] == multiplier) {
                        nEdge[i]++;
                        nEdge[j]++;
                    }
                }
            }

            double[] eigs = new EigenDecomposition(new Array2DRowRealMatrix(this.prec)).getRealEigenvalues();
            double eig = eigs[0];
            for (double value : eigs) {
                if (value < eig) eig = value;
            }
            // P = 50: around 0.2
            double multiplier2 = 1.5 / (Math.abs(eig) + 0.1 + 0.5);

            for (int i = 0; i < this.P; i++) {
                for (int j = 0; j < this.P; j++) {
                    if (i == j) {
                        this.prec[i][j] = 1;
                    } else {
                        this.prec[i][j] = this.prec[i][j] != 0 ? multiplier2 : 0;
                    }
                }
            }

        } else if (covTypeSim.equals("AR1")) {
            this.prec[0][0] = 1;
            nEdge[0] = 1;
            for (int i = 1; i < P; i++) {
                this.prec[i][i] = 1;
                this.prec[i][i - 1] = 0.5;
                this.prec[i - 1][i] = 0.5;
                nEdge[i] = 2;
            }
            nEdge[P - 1] = 1;
        }  else if (covTypeSim.equals("AR2")) {
            this.prec[0][0] = 1;
            nEdge[0] = 1;
            for (int i = 1; i < P; i++) {
                this.prec[i][i] = 1;
                this.prec[i][i - 1] = 0.5;
                this.prec[i - 1][i] = 0.5;
                nEdge[i] = 2;
            }
            nEdge[P - 1] = 1;
        } else if (covTypeSim.equals("Hetero")) {
            for (int i = 0; i < P; i++) {
                this.prec[i][i] = 1 / (i + 1.0);
            }
        }else if(covTypeSim.equals("Star")){
            for (int i = 0; i < P; i++) {
                this.prec[i][i] = 1;
            }
            for(int i = 1; i < P; i++){
                this.prec[0][i] = 0.1;
                this.prec[i][0] = 0.1;
            }
        }else if(covTypeSim.equals("Dense")){
            double[][] Ip = new double[this.P][this.P];
            for(int i = 0; i < this.P; i++) Ip[i][i] = 1;
            RealMatrix Ip_mat = new Array2DRowRealMatrix(Ip);
            RealMatrix prec0 = MathUtil.SampleWishart(this.P + 2.0, Ip_mat, rngEngine);
            double[][] cov0 = new LUDecomposition(prec0).getSolver().getInverse().getData();
            double[] diag = new double[P];
            for(int i = 0; i < this.P; i++){
                diag[i] = cov0[i][i];
            }
            for(int i = 0; i < this.P; i++) {
                for (int j = 0; j < this.P; j++) {
                    cov0[i][j] /= Math.sqrt(diag[i] * diag[j]);
                }
            }
            this.prec = new LUDecomposition(new Array2DRowRealMatrix(cov0)).getSolver().getInverse().getData();
        }else if(covTypeSim.equals("Random2")){
            int P1 = this.P / 2;
            int P2 = this.P - P1;
            double[][] Ip1 = new double[P1][P1];
            for(int i = 0; i < P1; i++) Ip1[i][i] = 1;
            double[][] Ip2 = new double[P2][P2];
            for(int i = 0; i < P2; i++) Ip2[i][i] = 1;

            RealMatrix Ip_mat1 = new Array2DRowRealMatrix(Ip1);
            RealMatrix Ip_mat2 = new Array2DRowRealMatrix(Ip2);

            RealMatrix prec1 = MathUtil.SampleWishart(P1 + 2.0, Ip_mat1, rngEngine);
            RealMatrix prec2 = MathUtil.SampleWishart(P2 + 2.0, Ip_mat2, rngEngine);

            RealMatrix prec0 = new Array2DRowRealMatrix(new double[this.P][this.P]);
            for(int i = 0; i < P1; i++){
                for(int j = 0; j < P1; j++){
                    prec0.setEntry(i, j, prec1.getEntry(i, j));
                }
            }
            for(int i = P1; i < P1 + P2; i++){
                for(int j = P1; j < P1 + P2; j++){
                    prec0.setEntry(i, j, prec2.getEntry(i - P1, j - P1));
                }
            }

            double[][] cov0 = new LUDecomposition(prec0).getSolver().getInverse().getData();
            double[] diag = new double[P];
            for(int i = 0; i < this.P; i++){
                diag[i] = cov0[i][i];
            }
            for(int i = 0; i < this.P; i++) {
                for (int j = 0; j < this.P; j++) {
                    cov0[i][j] /= Math.sqrt(diag[i] * diag[j]);
                }
            }
            this.prec = new LUDecomposition(new Array2DRowRealMatrix(cov0)).getSolver().getInverse().getData();
        }else{
            System.out.println("Wrong Covariance type");
        }

        RealMatrix prec = new Array2DRowRealMatrix(this.prec);
        this.cov = new LUDecomposition(prec).getSolver().getInverse().getData();
        // change covariance to correlation
        double[] diag = new double[P];
        for(int i = 0; i < this.P; i++){
            diag[i] = this.cov[i][i];
        }
        for(int i = 0; i < this.P; i++){
            for(int j = 0; j < this.P; j++){
                this.prec[i][j] = this.prec[i][j] * Math.sqrt(diag[i] * diag[j]);
                this.cov[i][j] = this.cov[i][j] / Math.sqrt(diag[i] * diag[j]);
            }
        }

        // this.prec = new LUDecomposition(new Array2DRowRealMatrix(this.cov)).getSolver().getInverse().getData();

        System.out.println("Number of edges per each node");
        System.out.println(Arrays.toString(nEdge));
    }

    public void set_sim(Latent_model model,
                        MersenneTwister rngEngin,
                        Random rand, String covType, double miss_rate,
                        boolean informative, boolean misspecified, int seed) {
        set_sim(model, rngEngin, rand, covType, miss_rate, informative, misspecified, 0, false, seed);
    }

    /**
     *  Set model parameters to simulated values
     *
     * @param model the model to set
     * @param rngEngin
     * @param rand
     * @param covType Type of covariance matrix "PX" or "SSSL" or "Glasso"
     * @param miss_rate percentage of missing data
     * @param informative whether to set the priors for graph structure informative
     * @param misspecified whether the probs are misspecified
     * @param nContinuous number of continuous variables to include
     * @param transform whether or not the continuous variables are transformed
     *
     *        alphaDirichlet parameter for dirichlet
     */

    public void set_sim(Latent_model model,
                        MersenneTwister rngEngin,
                        Random rand, String covType, double miss_rate,
                        boolean informative, boolean misspecified,
                        int nContinuous, boolean transform, int seed) {
        set_sim(model, rngEngin, rand, covType, miss_rate, informative, misspecified, nContinuous, transform, 0.0, seed);
    }
    public void set_sim(Latent_model model,
                        MersenneTwister rngEngin,
                        Random rand, String covType, double miss_rate,
                        boolean informative, boolean misspecified,
                        int nContinuous, boolean transform, double alphaDirichlet, int seed){
        set_sim(model, rngEngin, rand, covType, miss_rate, informative, misspecified, nContinuous, transform, 0.0, seed, false);
    }
    public void set_sim(Latent_model model,
                        MersenneTwister rngEngin,
                        Random rand, String covType, double miss_rate,
                        boolean informative, boolean misspecified,
                        int nContinuous, boolean transform, double alphaDirichlet, int seed, boolean use_probbase) {
        int N = this.N;
        int P = this.P;
        int G = this.G;
        double[] mu0 = new double[P];

        /**  simulate data **/
        MultivariateNormalDistribution rngMN_PP = new MultivariateNormalDistribution(rngEngin, mu0, this.cov);
        this.Z = new double[N][P];
        for(int i = 0; i < N; i++){
            double[] corr_error = rngMN_PP.sample();
            for(int j = 0; j < P; j++){
                this.Z[i][j] = corr_error[j];
            }
        }

        /** simulate cut-off **/
        NormalDistribution rngN = new NormalDistribution(rngEngin, 0, 1);

        Double[][] cutoff = new Double[G][P];
        Double[][] cutoff_out = new Double[G][P];
        int[] type = new int[P];
        for(int i = 0; i < nContinuous; i++) {
            type[i] = 0;
        }
        for(int i = nContinuous; i < P; i++){
            type[i] = 1;
        }


        List<Integer> shuffle = new ArrayList<Integer>();
        for(int i = 0; i < this.P; i++) shuffle.add(i);

        // initiate cut-off values
        if(misspecified) {
//            System.out.print("*");
//            double[] cutoff1 = new double[G * P];
//            double[] cutoff2 = new double[G * P];
//            for(int i = 0; i < cutoff1.length; i++){
//                double tmp = 1 - rand.nextDouble();//Math.exp(rand.nextDouble() * (Math.log(0.9) - Math.log(0.1)) + Math.log(0.1));
//                cutoff1[i] = rand.nextDouble() * 2 - 1;//new NormalDistribution().inverseCumulativeProbability(tmp);
//            }
//            Arrays.sort(cutoff1);
//            double[] cutoff2_base = new double[5];
//            for(int i = 0; i < cutoff2_base.length; i++){
//                double tmp = 1 - rand.nextDouble();//Math.exp(rand.nextDouble() * (Math.log(0.9) - Math.log(0.1)) + Math.log(0.1));
//                cutoff2_base[i] = rand.nextDouble() * 1 - 0.5;//new NormalDistribution().inverseCumulativeProbability
//                // (tmp);
//            }
//
//            for(int i = 0; i < cutoff2.length; i++) cutoff2[i] = cutoff2_base[rand.nextInt(cutoff2_base.length)];
//            Arrays.sort(cutoff2);
//
//            for (int g = 0; g < this.G; g++) {
//                Collections.shuffle(shuffle);
//                for (int i = 0; i < this.P; i++) {
//                    int j = shuffle.get(i);
//                    int k = rand.nextInt(cutoff1.length);
//                    cutoff[g][j] = cutoff1[k];
//                    cutoff_out[g][j] = cutoff2[k];
//                }
//            }

            // New misspecified prior
            for(int g = 0; g < this.G; g++) {
                for (int i = 0; i < this.P; i++) {
                    cutoff[g][i] = rand.nextDouble() * 2 - 1;
//                    cutoff_out[g][i] = cutoff[g][i] * 2;
                    cutoff_out[g][i] = cutoff[g][i] * cutoff[g][i] * (cutoff[g][i] > 0 ? 1 : -1);
//                    cutoff[g][i] = (rand.nextDouble() * 1 + 1) * (rand.nextDouble() > 0.8 ? 1 : -1)* (rand.nextDouble
//                            () > 0.5 ? 1 : 0);
//                    cutoff_out[g][i] = cutoff[g][i] / 2;
                }
            }

        }else if(use_probbase){
            // MAKING IT USING VA PROBBASE
//            double[] probbase = {0.9999, 0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
//                    0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00005, 0.000001};
            for(int g = 0; g < this.G; g++){
                for (int i = 0; i < this.P; i++) {
                    double rnd = new Random().nextDouble() * (Math.log(0.99) - Math.log(0.01)) + Math.log(0.01);
//                    double rnd = new Random().nextDouble() * (Math.log(0.9999) - Math.log(0.0001)) + Math.log(0.0001);
                    rnd = 1 - Math.exp(rnd);
                    cutoff[g][i] = new NormalDistribution().inverseCumulativeProbability(rnd);
                    cutoff_out[g][i] = cutoff[g][i];
                }
            }

        }else{
            for(int g = 0; g < this.G; g++) {
                for (int i = 0; i < this.P; i++) {
                    cutoff[g][i] = rand.nextDouble() * 2 - 1;
                    cutoff_out[g][i] = cutoff[g][i];
                }
            }
        }


        // set mean to the cut-off values
        this.mean = new double[this.G][this.P];
        for(int g = 0; g < this.G; g++){
            for(int i = 0; i < this.P; i++){
                // todo: initialize mean for categorical variables
                this.mean[g][i] = type[i] == 0 ? 0 : cutoff_out[g][i];
            }
        }


        // set equal number of group sizes
        this.membership = new int[N];
        this.prob = new double[G];
        if(alphaDirichlet == 0){
            for(int i = 0; i < this.N; i++){
                membership[i] = i % G;
            }
            for(int g = 0; g < G; g++) this.prob[g] = 1 / (G + 0.0);
        }else{
            DoubleMersenneTwister rngEngine = new DoubleMersenneTwister(seed);
            double[] alphas = new double[G];
            for(int g = 0; g < G; g++) alphas[g] = alphaDirichlet;
            Gamma rngG = new Gamma(1.0, 1.0, rngEngine);
            double[] probs = MathUtil.sampleDirichlet(alphas, rngG);
            for(int g = 0; g < G; g++) this.prob[g] = probs[g];
            for(int i = 0; i < this.N; i++){
                membership[i] = MathUtil.discrete_sample(probs, rand.nextDouble());
            }
            System.out.println("Simulated CSMF: " + Arrays.toString(probs));
        }


        /** set X, note for continuous and categorical variables,
         *      the mean parameters need to be updated.
         */

        this.X = new double[N][P];
        double[][] deltahat = new double[G][P];
        for(int j = 0; j < this.P; j++){
            if(type[j] == 0){
                double[] tempMean = new double[this.G];
                for(int i = 0; i < this.N; i++){
                    this.X[i][j] = this.Z[i][j];
                    if(transform){
                        this.X[i][j] = this.X[i][j] > 0 ? Math.pow(this.X[i][j], 1.0/3.0) : -1 * Math.pow(-this
                                .X[i][j], 1.0/3.0);
                    }
                    tempMean[this.membership[i]] += this.X[i][j];
                }

            }else if(type[j] == 1){
                double[] tmp = new double[G];
                double[] totaltmp = new double[G];
                for(int i = 0; i < this.N; i++){
                    this.X[i][j] = this.Z[i][j] > cutoff[membership[i]][j] ? 1 : 0;
                    tmp[membership[i]] += this.X[i][j];
                    totaltmp[membership[i]] += 1;
                }
                for(int g = 0; g < this.G; g++){
//                    if(tmp[g] == totaltmp[g]){
//                        deltahat[g][j] = rngN.inverseCumulativeProbability(1 - 1 / (1.0 + this.N));
//                    }else if(tmp[g] > 0){
//                        deltahat[g][j] = rngN.inverseCumulativeProbability(tmp[g] / (totaltmp[g] + 0.0));
//                    }else{
//                        deltahat[g][j] = rngN.inverseCumulativeProbability(1 / (1.0 + this.N));
//                    }
                    deltahat[g][j] = -cutoff_out[g][j];
                }
            }else if(type[j] == 2){
                // todo : categorical case simulation
            }
        }

        // data contamination: if missing, make it -Inf...
        for(int i = 0; i < this.N; i++){
            for(int j = 0; j < this.P; j++){
                this.X[i][j] = rand.nextDouble() > miss_rate ? this.X[i][j] : -Double.MAX_VALUE;
            }
        }

        /**  standard hyper-priors **/
        double sigma_px = 1;
        double v0 = 0.01;
        double v1 = 1;
        double lambda = 1;
        double[][] alpha = new double[P][P];
        double[][] beta = new double[P][P];

        double sparsity = 0;
        for(int i = 0; i < P; i++){
            for(int j = 0; j < P; j++){
                if(i != j) sparsity += (prec[i][j] == 0) ? 0 : 1;
            }
        }
        sparsity /= (P * (P - 1));
        if(sparsity > 0.1){
            sparsity = 0.1;
        }
//        System.out.println(new LUDecomposition(new Array2DRowRealMatrix(cov)).getDeterminant());
        /**  PX **/
        if(covType.equals("PX")){
            model.cov_px.setTruth(cov, prec);

        /**  SSSL  **/
        }else if(covType.equals("SSSL")){
            for(int i = 0; i < P; i++){
                for(int j = 0; j < P; j++){
                    alpha[i][j] = 1;
                }
            }
            for(int i = 0; i < P; i++){
                for(int j = 0; j < P; j++){
                    beta[i][j] =  alpha[i][j] * (1 - 1.0 * sparsity) / (1.0 * sparsity);
                }
            }
            // pretend we know the right prior
            for(int i = 0; i < P; i++){
                for(int j = 0; j < P; j++){
                    if(this.prec[i][j] != 0 & informative){
                        alpha[i][j] += 1;
                    }else if(informative){
                        beta[i][j] += 1;
                    }
                }
            }

            model.cov_sssl.setTruth(this.cov, this.prec);
            model.cov_sssl.sethyper(G, v0, v1, lambda, alpha, beta);

        /**  GLasso  **/
        }else if(covType.equals("GLASSO")){
            // todo : set hyper parameters for glasso model here
            model.cov_glasso.setTruth(cov, prec);
        }

        model.data.N = this.N;
        model.data.init_group(this.membership);
        model.data.ID = new String[this.N];
        model.data.data = new double[this.N][this.P];
        model.data.latent = new double[this.N][this.P];
        model.data.expanded = new double[this.N][this.P];
        model.data.Delta = new double[G][this.P];
        model.data.Delta_expanded = new double[G][this.P];
        for(int i = 0; i < N; i++){
            for(int j = 0; j < P; j++){
                if(type[j] == 0){
                    model.data.latent[i][j] = this.X[i][j] == -Double.MAX_VALUE ? 0: this.X[i][j];
                }else{
                    model.data.latent[i][j] = this.X[i][j] == -Double.MAX_VALUE ? 0: -1 + 2 * this.X[i][j];
                }
                model.data.data[i][j] = this.X[i][j];
                model.data.expanded[i][j] = model.data.latent[i][j];
            }
            model.data.ID[i] = "id" + i;
        }
        for(int j = 0; j < P; j++){
            for(int g = 0; g < G; g++){
                model.data.mu[g][j] = deltahat[g][j];
                model.data.Delta[g][j] = deltahat[g][j];
            }
            model.data.type[j] = type[j];
        }
        if(alphaDirichlet > 0){
            for(int g = 0; g < this.G; g++){
                model.true_prob[g] = this.prob[g];
            }
        }
    }

}