package sampler;

import cern.jet.random.tdouble.Exponential;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
import math.MultivariateNormal;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.random.MersenneTwister;
import util.EvalUtil;
import util.MathUtil;
import util.Simulator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.*;

/**
 * This is a higher-level implementation of sampler.Latent_model
 *      Allows different classes of data to have different prior means,
 *      The correlation structure is estimated to be the same.
 *      Add also the NB classifier
 *
 *  TODO: connecting with InSilicoVA
 *
 * Created by zehangli on 12/2/16.
 */
public class Latent_classifier extends Latent_model{

    double[] prior_prob;
    double[] alpha; // parameter for the Dirichlet prior
    double[] mu_theta; // parameter for the normal prior
    double[] var_theta;// parameter for the normal prior
    double[][] lower;
    double[][] upper;
    boolean domain_computed;
    MultivariateNormal mvn;
    int N_test;
    ArrayList<Integer>[] nonmissingbinary;
    public boolean keeptruth;
    public int[] true_membership;
    public boolean informative_prior;
    public boolean[][] impossible;

    public void setanneal(boolean anneal){this.anneal = anneal;}


    public Latent_classifier(int Nitr, int burnin, int thin, int N, int P, int G, String covType){
        super(Nitr, burnin, thin, N, P, G, covType);
    }

    public Latent_classifier(int Nitr, int N, int P, int G, String covType){
        super(Nitr, N, 0, P, G, covType);
    }
    public Latent_classifier(int Nitr, int N, int N_test, int P, int G, String covType){super(Nitr, N, N_test, P, G, covType);}
    public Latent_classifier(int Nitr, int burnin, int thin, int N,  int N_test, int P, int G, String covType){super(Nitr, burnin, thin, N, N_test, P, G, covType);}

    public void findimpossible(double shrink){
        double threshold = -4.3 / shrink; // todo: change this from hard coding
        impossible = new boolean[this.N][this.G];
        int count1 = 0;
        int count2 = 0;
        for(int i = 0; i < this.N; i++){
            boolean affect = false;
            for(int j = 0; j < this.P; j++){
                for(int k = 0; k < this.G; k++){
                    if(this.data.type[j] == 1 & this.data.mu[k][j] < threshold & this.data.data[i][j] == 1){
                        this.impossible[i][k] = true;
                        count1 ++;
                        affect = true;
                    }
                }
            }
            if(affect) count2++;
        }
        System.out.println(count1 + " instance of impossible causes identified: affected death: " + count2);
    }

    public void findimpossible(int[][] impossible){
        int count1 = 0;
        int count2 = 0;
        for(int i = 0; i < this.N_test; i++){
            boolean affect = false;
                for(int k = 0; k < this.G; k++){
                    if(impossible[i][k] == 1 & this.impossible[i][k] == false){
                        this.impossible[i][k] = true;
                        count1 ++;
                        affect = true;
                    }
            }
            if(affect) count2++;
        }
        System.out.println(count1 + " instance of impossible causes updated from physician: affected death: " + count2);
    }


    public void findimpossible(){
        impossible = new boolean[this.N][this.G];
        for(int i = 0; i < this.N; i++){
            for(int j = 0; j < this.P; j++){
                for(int k = 0; k < this.G; k++){
                        this.impossible[i][k] = false;
                }
            }
        }
    }

    // initialize uniform prior
    public void init_prior(int G, int N_test, double alpha0, Random rand, Double var0, boolean informative_prior){
        this.true_prob = new double[G];
        this.prior_prob = new double[G];
        this.post_prob = new double[G];
        this.post_prob_theta_transform = new double[G];
        this.prior_mean_theta_transform = new double[G];
        this.alpha = new double[G];
        this.mu_theta = new double[G];
        this.var_theta = new double[G];
        this.var0 = var0;
        this.informative_prior = informative_prior;
        for(int i = 0; i < G; i++){
            this.var_theta[i] = var0;
            this.prior_prob[i] = 1 / (G + 0.0);
            this.post_prob[i] = 1 / (G + 0.0);
            // this is of length p-1, first fix at 1, removed from here
            this.post_prob_theta_transform[i] = Math.log(this.post_prob[i] / this.post_prob[0]);
            this.prior_mean_theta_transform[i] = Math.log(this.post_prob[i] / this.post_prob[0]);
            this.alpha[i] = alpha0;
        }
        this.domain_computed = false;
        this.mvn = MultivariateNormal.DEFAULT_INSTANCE;
        this.test_case = new boolean[N];
        this.N_test = N_test;
        this.post_prob_draw = new double[this.G][this.Nout][this.N_test];
        this.post_prob_pop = new double[this.G][this.Nout];
        this.penalty_mat = new double[this.P][this.P];
    }
    // initialize uniform prior
    public void init_prior(int G, int N_test, double alpha0, double[][] mat, double var0, boolean informative_prior){
        this.true_prob = new double[G];
        this.prior_prob = new double[G];
        this.post_prob = new double[G];
        this.alpha = new double[G];
        this.var0 = var0;
        this.var_theta = new double[G];
        this.informative_prior = informative_prior;
        for(int i = 0; i < G; i++){
            this.var_theta[i] = var0;
            this.prior_prob[i] = 1 / (G + 0.0);
            this.post_prob[i] = 1 / (G + 0.0);
            this.alpha[i] = alpha0;
        }
        this.domain_computed = false;
        this.mvn = MultivariateNormal.DEFAULT_INSTANCE;
        this.test_case = new boolean[N];
        this.N_test = N_test;
        this.post_prob_draw = new double[this.G][this.Nout][this.N_test];
        this.post_prob_pop = new double[this.G][this.Nout];
        this.penalty_mat = new double[this.P][this.P];
        for(int i = 0; i < mat.length; i ++){
            for(int j = 0; j < mat.length; j++){
                this.penalty_mat[i][j] = mat[i][j];
            }
        }
    }

    // initialize with uninformative prior but informative initialization
    public void init_prior(int G, int N_test, double alpha0, double[][] mat, double var0, boolean informative_prior,
                           double[] csmf_init){
        this.true_prob = new double[G];
        this.prior_prob = new double[G];
        this.post_prob = new double[G];
        this.alpha = new double[G];
        this.var0 = var0;
        this.var_theta = new double[G];
        this.informative_prior = informative_prior;
        for(int i = 0; i < G; i++){
            this.var_theta[i] = var0;
            this.prior_prob[i] = csmf_init[i];
            this.post_prob[i] = csmf_init[i];
            this.alpha[i] = alpha0; // note this does not change prior, only changes initialization
        }
        this.domain_computed = false;
        this.mvn = MultivariateNormal.DEFAULT_INSTANCE;
        this.test_case = new boolean[N];
        this.N_test = N_test;
        this.post_prob_draw = new double[this.G][this.Nout][this.N_test];
        this.post_prob_pop = new double[this.G][this.Nout];
        this.penalty_mat = new double[this.P][this.P];
        for(int i = 0; i < mat.length; i ++){
            for(int j = 0; j < mat.length; j++){
                this.penalty_mat[i][j] = mat[i][j];
            }
        }
    }

     public void update_prior_prob(double[] csmf){
        this.prior_prob = new double[G];
        this.post_prob = new double[G];
        this.mu_theta = new double[G];
        for(int i = 0; i < csmf.length; i++){
            this.prior_prob[i] += csmf[i] * G;
            this.post_prob[i] +=  csmf[i] * G;
        }


        this.post_prob_theta_transform = new double[this.prior_prob.length];
        this.prior_mean_theta_transform = new double[this.prior_prob.length];
        for(int i = 0; i < this.prior_prob.length; i++){
            this.post_prob_theta_transform[i] =Math.log(this.prior_prob[i] / this.prior_prob[0]);
            this.prior_mean_theta_transform[i] =Math.log(this.prior_prob[i] / this.prior_prob[0]);
            this.mu_theta[i] = Math.log(prior_prob[i]);
        }
        for(int i = 0; i < G; i++){
            this.alpha[i] *= this.prior_prob[i];
        }
    }
    public void update_prior_prob(int[] priorlabel){
        this.prior_prob = new double[G];
        this.post_prob = new double[G];
        this.mu_theta = new double[G];
        for(int i = 0; i < priorlabel.length; i++){
            this.prior_prob[priorlabel[i]] += 1 / (priorlabel.length + G + 0.0);
            this.post_prob[priorlabel[i]] += 1 / (priorlabel.length + G + 0.0);
        }
        for(int i = 0; i < this.prior_prob.length; i++){
            this.prior_prob[i] += 1 / (priorlabel.length + G + 0.0);
            this.post_prob[i] += 1 / (priorlabel.length + G + 0.0);
        }
        this.post_prob_theta_transform = new double[this.prior_prob.length];
        this.prior_mean_theta_transform = new double[this.prior_prob.length];
        for(int i = 0; i < this.prior_prob.length; i++){
            this.post_prob_theta_transform[i] =Math.log(this.prior_prob[i] / this.prior_prob[0]);
            this.prior_mean_theta_transform[i] =Math.log(this.prior_prob[i] / this.prior_prob[0]);
            this.mu_theta[i] = Math.log(prior_prob[i]);
        }
        for(int i = 0; i < G; i++){
            this.alpha[i] *= this.prior_prob[i];
        }
    }
    public void update_prior_prob(){
        this.prior_prob = new double[G];
        this.post_prob = new double[G];
        this.mu_theta = new double[G];

        for(int i = 0; i < this.G; i++){
            this.prior_prob[i] += 1 / (G + 0.0);
            this.post_prob[i] += 1 / (G + 0.0);
        }
        this.post_prob_theta_transform = new double[this.prior_prob.length];
        this.prior_mean_theta_transform = new double[G];
        for(int i = 0; i < this.prior_prob.length; i++){
            this.post_prob_theta_transform[i] = Math.log(this.prior_prob[i] / this.prior_prob[0]);
            this.prior_mean_theta_transform[i] = Math.log(this.prior_prob[i] / this.prior_prob[0]);
        }
        for(int i = 0; i < G; i++){
            this.prior_prob[i] += 1 / (this.G + 1.0);
            this.alpha[i] *= this.prior_prob[i];
        }
    }

    public void update_group_prob(double[][] pnb, Random rand, Gamma rngG, boolean same_pop, double temp) {
        update_group_prob(pnb, rand, rngG, same_pop, temp, true);
    }
    public void update_group_prob(double[][] pnb0, Random rand, Gamma rngG, boolean same_pop, double temp, boolean
            update_assignments){

        double[][] pnb = new double[pnb0.length][pnb0[0].length];
        for(int i = 0; i < pnb.length; i++){
            for(int j = 0; j < pnb[0].length; j++){
                pnb[i][j] = pnb0[i][j];
            }
        }
        /** Dirichlet prior **/
        if(this.Dirichlet){
            double[] counts = new double[this.G];
            int[] removed = new int[this.G];
            double sumalpha = 0;
            for(int j = 0; j < this.G; j++) sumalpha += this.alpha[j];
            int nn = this.N;
            if(same_pop){
                for(int i = 0; i < this.N; i++){
                    if(this.data.membership[i] < G){
                        counts[this.data.membership[i]] += 1;
                    }else if(this.data.membership_test[i] < G){
                        counts[this.data.membership_test[i]] += 1;
                    }
                }
            }else{
                for(int i = 0; i < this.N_test; i++){
                    if(this.data.membership_test[i] < G) counts[this.data.membership_test[i]] += 1;
                }
                nn = this.N_test;
            }

            if(update_assignments) {
                for (int i = 0; i < pnb.length; i++) {
                    for (int j = 0; j < this.G; j++) {
                        if (this.post_prob[j] > 0) pnb[i][j] /= Math.pow(this.post_prob[j], 1);
//                    pnb[i][j] = Math.pow(pnb[i][j], temp);
//                    pnb[i][j] *= (counts[j]-(this.data.membership_test[i]==j ? 1:0) * temp) + alpha[j];
//                    pnb[i][j] /= (nn * temp + sumalpha);
                        pnb[i][j] *= (counts[j] - (this.data.membership_test[i] == j ? 1 : 0)) + alpha[j];
                        pnb[i][j] /= (nn + sumalpha);
                        pnb[i][j] = Math.pow(pnb[i][j], temp);
                    }

                    int pick = MathUtil.discrete_sample(pnb[i], rand.nextDouble());
//            //todo: warning warning!! Only usable when test data is at beginning of the data!!!
                    if (this.data.membership_test[i] < G)
                        removed[this.data.membership_test[i]] += pick == this.data.membership_test[i] ? 0 : 1;
                    if (this.data.membership_test[i] < G) {
                        counts[this.data.membership_test[i]] -= 1;
                        counts[pick]++;
                    }
                    this.data.membership_test[i] = pick;

                }
            }

            if(this.keeptruth){
                double right = 0;
                for(int i = 0; i < pnb.length; i++){
                    right += this.data.membership_test[i] == this.true_membership[i] ? 1 : 0;
                }
                System.out.println("Correct case: " + right + " Accuracy: " + right / (pnb.length + 0.0));
            }
            System.out.println("Number in group: " +Arrays.toString(counts));
            System.out.println("Number left group: " + Arrays.toString(removed));

            // Sample Dirichlet
            for(int i = 0; i < G; i++) counts[i] = counts[i] * temp + this.alpha[i] * temp;
            double[] tmp = MathUtil.sampleDirichlet(counts, rngG);
            for(int i = 0; i < G; i++) this.post_prob[i] = tmp[i];
            System.out.println(Arrays.toString(this.post_prob));

            /** Logisitic Normal prior **/
        }else {
            double[] newalpha = new double[this.alpha.length];
            int[] removed = new int[this.G];
            if(update_assignments) {
                for (int i = 0; i < pnb.length; i++) {
                    if (temp < 1) {
                        for (int j = 0; j < this.G; j++) {
                            pnb[i][j] = Math.pow(pnb[i][j], temp);
                        }
                    }
                    int pick = MathUtil.discrete_sample(pnb[i], rand.nextDouble());

//            //todo: warning warning!! Only usable when test data is at beginning of the data!!!
                    if (this.data.membership_test[i] < G)
                        removed[this.data.membership_test[i]] += pick == this.data.membership_test[i] ? 0 : 1;
                    this.data.membership_test[i] = pick;
                    newalpha[pick]++;
                }
            }
            if (this.keeptruth) {
                double right = 0;
                for (int i = 0; i < pnb.length; i++) {
                    right += this.data.membership_test[i] == this.true_membership[i] ? 1 : 0;
                }
                System.out.println("Correct case: " + right + " Accuracy: " + right / (pnb.length + 0.0));
            }
            if (same_pop & this.N > this.N_test) {
                for (int i = this.N_test; i < this.N; i++) {
                    newalpha[this.data.membership[i]]++;
                }
            }
            System.out.println("Number in group: " + Arrays.toString(newalpha));
            System.out.println("Number left group: " + Arrays.toString(removed));
            if (informative_prior) {
                // resample mu assuming informative prior
                this.mu_theta[0] = 0;
                for (int i = 1; i < this.G; i++) {
                    double sd = Math.sqrt(1 / (1 / this.var_theta[i] + 1 / this.var0));
                    this.mu_theta[i] = sd * sd * (this.prior_mean_theta_transform[i] / this.var0
                            + this.post_prob_theta_transform[i] / this.var_theta[i]) + rand.nextGaussian() * sd;
                }
                double sumsquare = 0;
                for (int i = 0; i < this.G; i++)
                    sumsquare += Math.pow(this.post_prob_theta_transform[i] - this.mu_theta[i], 2);
                this.var_theta[0] = 1 / rngG.nextDouble((this.G - 2.0) / 2.0, 0.5 * sumsquare);
                this.var_theta[0] = this.var_theta[0] == 0 ? 1 : this.var_theta[0];
                for (int i = 1; i < this.G; i++) this.var_theta[i] = this.var_theta[0];
                System.out.println("Estimated mu_theta: " + Arrays.toString(mu_theta));
                System.out.println("Estiamted Var_theta: " + Arrays.toString(var_theta));
            } else {
                // resample mu assuming non informative prior
                this.mu_theta[0] = rand.nextGaussian() * Math.sqrt(this.var_theta[0] / (this.G - 1 + 0.0));
                for (int i = 1; i < this.G; i++)
                    this.mu_theta[0] += this.post_prob_theta_transform[i] / (this.G - 1 + 0.0);
                for (int i = 1; i < this.G; i++) this.mu_theta[i] = this.mu_theta[0];
                double sumsquare = 0;
                for (int i = 0; i < this.G; i++)
                    sumsquare += Math.pow(this.post_prob_theta_transform[i] - this.mu_theta[i], 2);
                this.var_theta[0] = 1 / rngG.nextDouble((this.G - 2.0) / 2.0, 0.5 * sumsquare);
                this.var_theta[0] = this.var_theta[0] == 0 ? 1 : this.var_theta[0];
                for (int i = 1; i < this.G; i++) this.var_theta[i] = this.var_theta[0];
                System.out.println("Estimated mu_theta: " + mu_theta[0]);
                System.out.println("Estiamted Var_theta: " + var_theta[0]);
            }

            double[] tmp = ESSsampler.sample(this.post_prob_theta_transform, 5, this.mu_theta, this.var_theta,
                    newalpha, rand, 100, true);
            double sum = 0;
            for (int i = 0; i < tmp.length; i++) {
                this.post_prob_theta_transform[i] = tmp[i];
                this.post_prob[i] = Math.exp(tmp[i]);
                sum += this.post_prob[i];
            }
            for (int i = 0; i < this.post_prob.length; i++) {
                this.post_prob[i] /= sum;
            }
            System.out.println(Arrays.toString(this.post_prob));
        }
    }


    public void initial_test_label(Random rand, Gamma rngG, NormalDistribution rngN,
                                   Exponential rngE, boolean same_pop, boolean random){
        if(random){
            int[] count = new int[G];
            double max;
            for(int i = 0; i < this.N_test; i++){
                this.data.membership_test[i] = rand.nextInt(G);
                count[this.data.membership_test[i]] ++;
            }
            System.out.println("counts of the initial classes:" + Arrays.toString(count));
        }else{
            initial_test_label(rand, rngG, rngN, rngE, same_pop);
        }
    }

    public void initial_test_label(Random rand, Gamma rngG, NormalDistribution rngN, Exponential rngE, boolean same_pop){
        //if(update_with_test){
        double[][] pnb = update_group(rand, rngG, rngN, rngE, true, true, same_pop, 1);
        int[] count = new int[G];
        double max;
        for(int i = 0; i < this.N_test; i++){
            max = pnb[i][0];
            this.data.membership_test[i] = 0;
            for(int g  = 0; g < this.G; g++){
                if(pnb[i][g] > max){
                    max = pnb[i][g];
                    this.data.membership_test[i] = g;
                }
            }
            for(int j = 0; j < this.P; j++){
                double jmin = (this.data.data[i][j] == 1 & this.data.type[j] == 1) ? 0 : -5;
                double jmax = (this.data.data[i][j] == 0 & this.data.type[j] == 1) ? 0 : 5;
                this.data.latent[i][j] = MathUtil.truncNormal(rand, rngN, rngE, this.data.Delta[this.data.membership_test[i]][j],
                            1, jmin, jmax, 5);
            }

            count[this.data.membership_test[i]] ++;
        }
        System.out.println("counts of the initial classes:" + Arrays.toString(count));
        if(this.keeptruth){
            double right = 0;
            for(int i = 0; i < pnb.length; i++){
                right += this.data.membership_test[i] == this.true_membership[i] ? 1 : 0;
            }
            System.out.println("Correct case: " + right + " Accuracy: " + right / (pnb.length + 0.0));
        }

    }

    public void keeptruth(){
        this.keeptruth = true;
        this.true_membership = new int[this.N];
        for(int i = 0; i < this.N; i++) this.true_membership[i] = this.data.membership[i];
    }

    @Override
    public double[][] update_group_marginal( Random rand, Gamma rngG, NormalDistribution rngN, Exponential rngE,
                                             boolean integrate, boolean
            NB, boolean
                                            same_pop, double temp, boolean updateprob) {
            return (update_group_by_draw_marginal(rand, rngG, rngN, rngE, NB, same_pop, temp, updateprob));
    }

    @Override
    public double[][] update_group( Random rand, Gamma rngG, NormalDistribution rngN, Exponential rngE,boolean integrate, boolean
            NB, boolean
            same_pop, double temp){
        return(update_group(rand, rngG, rngN, rngE, integrate, NB, same_pop, temp, false));
    }

    @Override
    public double[][] update_group( Random rand, Gamma rngG, NormalDistribution rngN, Exponential rngE, boolean integrate, boolean
            NB, boolean
            same_pop, double temp, boolean updateprob){
        if(!integrate){
            return(update_group_by_draw(rand, rngG, rngN, rngE, NB, same_pop, temp, updateprob));
        }
        double[][] out = new double[N_test][G];

        // only compute the domain once
        if(!this.domain_computed){
            nonmissingbinary = new ArrayList[this.N];
            for(int i = 0; i < this.N; i++){
                nonmissingbinary[i] = new ArrayList<>();
            }
            this.upper = new double[this.N][this.P_binary];
            this.lower = new double[this.N][this.P_binary];
            int counter = 0;
            for(int j = 0; j < this.P; j++){
                if(this.data.type[j] == 1){
                    // todo: categorical case: support?
                    for(int i = 0; i < this.N; i++){
                        if(this.data.data[i][j] == 1){
                            this.upper[i][counter] = Double.POSITIVE_INFINITY;
                            this.lower[i][counter] = 0;
                            nonmissingbinary[i].add(counter);
                        }else if(this.data.data[i][j] == 0){
                            this.upper[i][counter] = 0;
                            this.lower[i][counter] = -Double.POSITIVE_INFINITY;
                            nonmissingbinary[i].add(counter);
                        }else if(this.data.data[i][j] == -Double.MAX_VALUE){
                            this.upper[i][counter] = Double.POSITIVE_INFINITY;
                            this.lower[i][counter] = -Double.POSITIVE_INFINITY;
                        }
                    }
                    counter++;
                }
            }
            this.domain_computed = true;
        }

        RealMatrix cov = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        if(covType.equals("SSSL")){
            cov = this.cov_sssl.corr;
        }else if(covType.equals("GLASSO")){
            cov = this.cov_glasso.corr;
        }else if(covType.equals("PX")){
            cov = this.cov_px.corr;
        }

        RealMatrix cov_bb_i = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        RealMatrix cov_bb = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        RealMatrix cov_ab = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        if(cont_indices.length > 0){
            RealMatrix cov_aa = cov.getSubMatrix(binary_indices, binary_indices);
            cov_ab = cov.getSubMatrix(binary_indices, cont_indices);
            cov_bb = cov.getSubMatrix(cont_indices, cont_indices);
            cov_bb_i = new LUDecomposition(cov_bb).getSolver().getInverse();
            cov = cov_aa.add(cov_ab.transpose().preMultiply(cov_bb_i).preMultiply(cov_ab).scalarMultiply(-1));
        }


        double[][] mean_binary = new double[this.G][this.P_binary];
        double[][] mean_cont = new double[this.G][this.P - this.P_binary];
        for(int g = 0; g < this.G; g++){
            RealMatrix mean_tmp = new Array2DRowRealMatrix(this.data.Delta[g]);
            if(cont_indices.length > 0){
                mean_tmp = mean_tmp.getSubMatrix(cont_indices, new int[]{0});
            }
            mean_cont[g] = mean_tmp.getColumn(0);
            RealMatrix mean_tmp2 = new Array2DRowRealMatrix(this.data.Delta[g]).getSubMatrix(binary_indices, new
                    int[]{0});
            mean_binary[g] = mean_tmp2.getColumn(0);
        }
        double[] prob = new double[this.G];
        int counter = 0;
        for(int i = 0; i < this.N_test; i++){
            RealMatrix cont_obs = new Array2DRowRealMatrix(this.data.latent[i]);
            if(cont_indices.length > 0){
                cont_obs = cont_obs.getSubMatrix(cont_indices, new int[]{0});
            }
            if(this.test_case[i]){
                double sum = 0;
                // note nonmissing is defined on the list of binary variables only!!!
                // so it can only be subsetted for binary variable objects
                int[] nonmissing =  new int[nonmissingbinary[i].size()];
                for(int k = 0; k < nonmissing.length; k++) nonmissing[k] = nonmissingbinary[i].get(k);
                RealMatrix cov2 = cov.getSubMatrix(nonmissing, nonmissing);
                for(int g = 0; g < this.G; g++){
                    RealMatrix mean2 = new Array2DRowRealMatrix(mean_binary[g]);
                    if(cont_indices.length > 0) {
                        mean2 = cont_obs.add(new Array2DRowRealMatrix(mean_cont[g]).scalarMultiply(-1));
                        mean2 = mean2.preMultiply(cov_bb_i).preMultiply(cov_ab);
                        mean2 = mean2.add(new Array2DRowRealMatrix(mean_binary[g]));
                    }
                    mean2 = mean2.getSubMatrix(nonmissing, new int[]{0});
                    double[] lower2 = new Array2DRowRealMatrix(lower[i]).getSubMatrix
                            (nonmissing, new int[]{0})
                            .getColumn(0);
                    double[] upper2 = new Array2DRowRealMatrix(upper[i]).getSubMatrix(nonmissing, new int[]{0})
                            .getColumn(0);

                    RealVector meanvec = mean2.getColumnVector(0);
                    if(NB){
                        prob[g] = MathUtil.diagCDF(mean2.getColumn(0), cov2, lower2, upper2) * this.post_prob[g];
//                        double tmp = this.mvn.cdf(meanvec, cov2, lower2, upper2).cdf * this.post_prob[g];
//                        System.out.println(prob[g] + tmp);
                    }else{
                        prob[g] = this.mvn.cdf(meanvec, cov2, lower2, upper2).cdf * this.post_prob[g];
                    }
                    if(cont_indices.length > 0) {
                        prob[g] *= MathUtil.multivariate_normal_density(cont_obs, mean_cont[g], cov_bb, NB);
                    }
                    sum += prob[g];
                }
                if(sum == 0) sum = 1;
                for(int g = 0; g < this.G; g++){
                    out[counter][g] = prob[g] / sum;
                }
                counter ++;
                if(counter % 20 == 0) System.out.print(".");
            }
        }
        System.out.println("Finishing resampling membership");
        if(updateprob){
            this.update_group_prob(out, rand, rngG, same_pop, temp);
            System.out.println("Finishing resampling Fraction Distribution");
        }
        return(out);
    }

    public double[][] update_group_by_draw(Random rand, Gamma rngG,  NormalDistribution rngN, Exponential rngE, boolean NB, boolean same_pop, double
            temp,
                                                 boolean update_prob) {

        double[][] out = new double[N_test][G];

        RealMatrix cov = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        if(covType.equals("SSSL")){
            cov = this.cov_sssl.corr;
        }else if(covType.equals("GLASSO")){
            cov = this.cov_glasso.corr;
        }else if(covType.equals("PX")){
            cov = this.cov_px.corr;
        }
        LUDecomposition solver = new LUDecomposition(cov);
        RealMatrix invcorr = solver.getSolver().getInverse();
        double det = solver.getDeterminant();

        double[] prob = new double[this.G];
        int counter = 0;
        for(int i = 0; i < this.N_test; i++){
            if(this.test_case[i]){
                double sum = 0;

                // remove conditioning on missing variables
                int nonmissing = 0;
                for(int j=0; j < this.P; j++){
                    if(this.data.data[i][j] != - Double.MAX_VALUE)  nonmissing++;
                }
                double[] zerovec = new double[this.G];
                double maxlog = -Double.MAX_VALUE;
                if(nonmissing == 0){
                    for(int g = 0; g < this.G; g++) {
                        double add = MathUtil.multivariate_normal_logdensity(this.data.latent[i], this.data.Delta[g],
                                invcorr, det);
                        prob[g] += add;
                        maxlog = Math.max(prob[g], maxlog);
                    }
                }else{
                    double[] x = new double[nonmissing];
                    double[][] meansub = new double[this.G][nonmissing];
                    int[] remain = new int[nonmissing];
                    int counter1 = 0;
                    for(int j = 0; j < this.P; j++){
                        if(this.data.data[i][j] != - Double.MAX_VALUE){
                            x[counter1] = this.data.latent[i][j];
                            for(int g = 0; g < this.G; g++) meansub[g][counter1] =  this.data.Delta[g][j];
                            remain[counter1] = j;
                            counter1 ++;
                        }
                    }
                    RealMatrix invcorrsub = new LUDecomposition(cov.getSubMatrix(remain, remain)).getSolver().getInverse();
                    for(int g = 0; g < this.G; g++){
                        if(this.impossible[i][g]){
                            prob[g] = 0; // just a place holder set to zero later
                            continue;
                        }
                        zerovec[g] = 1;
                        prob[g] = Math.log(this.post_prob[g]);
//                        RealMatrix covsub = cov.getSubMatrix(remain, remain);
//                        for(int k = 0; k < remain.length; k++) covsub.addToEntry(k, k, this.data.sd0[g][0]*this.data.sd0[g][0]);
//                        RealMatrix invcorrsub = new LUDecomposition(covsub).getSolver().getInverse();
//
                        double add = MathUtil.multivariate_normal_logdensity(x, meansub[g], invcorrsub, 1);
                        prob[g] += add;
                        maxlog = Math.max(prob[g], maxlog);
                    }
                }


                double expsum = 0;
                for(int g = 0; g < this.G; g++){
                    if(zerovec[g] != 0) expsum += Math.exp(prob[g] - maxlog);
                }

                for(int g = 0; g < this.G; g++){
                    out[counter][g] = zerovec[g] == 0 ? 0 : Math.exp(prob[g] - maxlog - Math.log(expsum));
                }
                counter ++;
                if(counter % 20 == 0) System.out.print(".");
            }
        }
        System.out.println("Finishing resampling membership");
       if(update_prob) this.update_group_prob(out, rand, rngG, same_pop, temp);
        System.out.println("Finishing resampling Fraction Distribution");
        return(out);
    }
    public double[][] update_group_by_draw_int(Random rand, Gamma rngG, boolean NB, boolean same_pop, double temp,
                                           boolean update_prob) {

        double[][] out = new double[N_test][G];

        RealMatrix cov = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        if(covType.equals("SSSL")){
            cov = this.cov_sssl.corr;
        }else if(covType.equals("GLASSO")){
            cov = this.cov_glasso.corr;
        }else if(covType.equals("PX")){
            cov = this.cov_px.corr;
        }

        List<Integer> onetopbin = new ArrayList<>();
        for (int i : binary_indices) {
            onetopbin.add(i);
        }
        double[] prob = new double[this.G];
        int counter = 0;
        for(int i = 0; i < this.N_test; i++){
            double maxlog = -Double.MAX_VALUE;
            if(this.test_case[i]){
                double sum = 0;

                int p_marg = 40;
                int p_cond = this.P - p_marg;
                int[] marg_indices = new int[p_marg];
                int[] latent_indices = new int[p_cond];

                Collections.shuffle(onetopbin);
                int mtmp = 0;
                int ctmp = 0;
                for(int j : this.binary_indices){
                    if(mtmp < p_marg & this.data.data[i][j] != - Double.MAX_VALUE){
                        marg_indices[mtmp] = onetopbin.get(j);
                        mtmp++;
                    }else{
                        latent_indices[ctmp] = onetopbin.get(j);
                        ctmp++;
                    }
                }
                for(int j = 0; j < this.cont_indices.length; j++){
                    latent_indices[i + this.binary_indices.length - p_marg] =
                            this.cont_indices[j];
                }


                // update conditional covariance matrix for marginalized variables
                RealMatrix cov_aa = cov.getSubMatrix(marg_indices, marg_indices);
                RealMatrix cov_ab = cov.getSubMatrix(marg_indices, latent_indices);
                RealMatrix cov_bb = cov.getSubMatrix(latent_indices, latent_indices);
                RealMatrix cov_bb_i = new LUDecomposition(cov_bb).getSolver().getInverse();
                RealMatrix cov1 = cov_aa.add(cov_ab.transpose().preMultiply(cov_bb_i).preMultiply(cov_ab).scalarMultiply(-1));


                // remove conditioning on missing variables
                int p_cond_not_missing = 0;
                for(int j : latent_indices){
                    if(this.data.data[i][j] != - Double.MAX_VALUE ) {
                        p_cond_not_missing++;
                    }
                }
                double[] zerovec = new double[this.G];

                // marginalized subvector
                double[][] mean_marg = new double[this.G][p_marg];
                double[] lower = new double[p_marg];
                double[] upper = new double[p_marg];
                int counter_tmp = 0;
                for(int j : marg_indices){
                    for(int g = 0; g < this.G; g++) mean_marg[g][counter_tmp] = this.data.Delta[g][j];
                    lower[counter_tmp] = this.data.data[i][j] > 0 ? 0 : -Double.POSITIVE_INFINITY;
                    upper[counter_tmp] = this.data.data[i][j] > 0 ? Double.POSITIVE_INFINITY : 0;
                    counter_tmp ++;
                }
                RealMatrix m1 = new Array2DRowRealMatrix(mean_marg);

                // conditional subvector
                double[] x2 = new double[p_cond_not_missing];
                double[][] m2 = new double[this.G][p_cond_not_missing];
                int[] remain = new int[p_cond_not_missing];
                int counter1 = 0;
                for(int j : latent_indices){
                    if(this.data.data[i][j] != - Double.MAX_VALUE){
                        x2[counter1] = this.data.latent[i][j];
                        for(int g = 0; g < this.G; g++) m2[g][counter1] =  this.data.Delta[g][j];
                        remain[counter1] = j;
                        counter1 ++;
                    }
                }
                RealMatrix cov2 = cov.getSubMatrix(remain, remain);
                RealMatrix invcov2 = new LUDecomposition(cov2).getSolver().getInverse();
                for(int g = 0; g < this.G; g++){
                    if(this.impossible[i][g]){
                        prob[g] = 0; // just a place holder set to zero later
                        continue;
                    }
                    zerovec[g] = 1;
                    prob[g] = Math.log(this.post_prob[g]);
                    double add1 = Math.log(this.mvn.cdf(m1.getRowVector(g), cov1, lower, upper).cdf);
                    double add2 = MathUtil.multivariate_normal_logdensity(x2, m2[g], invcov2, 1);
                    prob[g] += add1;
                    prob[g] += add2;
                    maxlog = Math.max(prob[g], maxlog);
                }

                double expsum = 0;
                for(int g = 0; g < this.G; g++){
                    if(zerovec[g] != 0) expsum += Math.exp(prob[g] - maxlog);
                }

                for(int g = 0; g < this.G; g++){
                    out[counter][g] = zerovec[g] == 0 ? 0 : Math.exp(prob[g] - maxlog - Math.log(expsum));
                }
                counter ++;
                System.out.print(".");
            }
        }
        System.out.println("Finishing resampling membership");
        if(update_prob) this.update_group_prob(out, rand, rngG, same_pop, temp);
        System.out.println("Finishing resampling Fraction Distribution");
        return(out);
    }

    public double[][] update_group_by_draw_MH(Random rand, Gamma rngG, NormalDistribution rngN, Exponential rngE, boolean NB,
                                              boolean
            same_pop,
                                           double temp) {
        RealMatrix cov = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        if(covType.equals("SSSL")){
            cov = this.cov_sssl.corr;
        }else if(covType.equals("GLASSO")){
            cov = this.cov_glasso.corr;
        }else if(covType.equals("PX")){
            cov = this.cov_px.corr;
        }
        LUDecomposition solver = new LUDecomposition(cov);
        RealMatrix invcorr = solver.getSolver().getInverse();
        double det = solver.getDeterminant();

        double[] prob = new double[this.G];
        int count = 0;
        for(int i = 0; i < this.N_test; i++){
            if(this.test_case[i]){
                double[] zerovec = new double[this.G];
                double maxlog = -Double.MAX_VALUE;
                int Ystar = rand.nextInt(G);
                while(this.impossible[i][Ystar]){
                    Ystar = rand.nextInt(G);
                }
                double[] Zstar = new double[this.P];
                for(int j = 0; j < this.P; j++){
                    if(this.data.type[j] == 0){
                        Zstar[j] = this.data.latent[i][j];
                    }else{
                        Zstar[j] = MathUtil.truncStdNormal1D(this.data.data[i][j], 0, rand, rngE);
                    }

                }
                double log_pi = MathUtil.multivariate_normal_logdensity(this.data.latent[i],this.data.Delta[this.data
                        .membership_test[i]], invcorr, det) + Math.log(this.post_prob[this.data.membership_test[i]]);
                double log_pistar = MathUtil.multivariate_normal_logdensity(Zstar,this.data.Delta[Ystar], invcorr,
                        det) + Math.log(this.post_prob[Ystar]);

                log_pi -= MathUtil.multivariate_normal_logdensity(this.data.latent[i]);
                log_pistar -= MathUtil.multivariate_normal_logdensity(Zstar);

                if(log_pistar - log_pi > Math.log(rand.nextDouble())){
                    this.data.membership_test[i] = Ystar;
                    for(int j = 0; j < this.P; j++){
                        this.data.latent[i][j] = Zstar[j];
                    }
                    count ++;
                }
                System.out.print(".");
            }
        }
        System.out.println("Finishing MH membership");
        System.out.println("Total switch: " + count);
        double[][] out = update_group_by_draw(rand, rngG, rngN, rngE, NB, same_pop, temp, true);
        this.update_group_prob(out, rand, rngG, same_pop, temp, false);
        System.out.println("Finishing resampling Fraction Distribution");
        return(out);
    }

    // this version use monte carlo approximation instead of pdf
    //
    public double[][] update_group_by_draw_marginal(Random rand, Gamma rngG, NormalDistribution rngN, Exponential rngE,
                                            boolean NB, boolean
            same_pop, double
            temp, boolean update_prob) {

        int M = 50;

        double[][] out = new double[N_test][G];

        RealMatrix cov = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        if(covType.equals("SSSL")){
            cov = this.cov_sssl.corr;
        }else if(covType.equals("GLASSO")){
            cov = this.cov_glasso.corr;
        }else if(covType.equals("PX")){
            cov = this.cov_px.corr;
        }

        RealMatrix cov_bb_i = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        RealMatrix cov_bb = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        RealMatrix cov_ab = new Array2DRowRealMatrix(this.alpha); // just a placeholder
        double[][] mean_cont = new double[this.G][this.P - this.P_binary];
        if(cont_indices.length > 0){
            // update conditional covariance matrix for binary variables
            RealMatrix cov_aa = cov.getSubMatrix(binary_indices, binary_indices);
            cov_ab = cov.getSubMatrix(binary_indices, cont_indices);
            cov_bb = cov.getSubMatrix(cont_indices, cont_indices);
            cov_bb_i = new LUDecomposition(cov_bb).getSolver().getInverse();
            cov = cov_aa.add(cov_ab.transpose().preMultiply(cov_bb_i).preMultiply(cov_ab).scalarMultiply(-1));
            for(int g = 0; g < this.G; g++){
                RealMatrix mean_tmp = new Array2DRowRealMatrix(this.data.Delta[g]);
                mean_cont[g] = mean_tmp.getSubMatrix(cont_indices, new int[]{0}).getColumn(0);
            }
        }

        for(int i = 0; i < this.N_test; i++){
            double maxlog = -Double.MAX_VALUE;
            double[] zerovec = new double[this.G];
            double[] sign = new double[this.P_binary];
            double[] min = new double[this.P_binary];
            double[] max = new double[this.P_binary];
            double[][] meansub = new double[this.G][this.P_binary];

            // count number of non missing dimensions
            int nonmissing = 0;
            for(int j=0; j < this.P_binary; j++){
                if(this.data.data[i][binary_indices[j]] != - Double.MAX_VALUE)  nonmissing++;
            }
            RealMatrix covsub = cov.getSubMatrix(0,0,0,0); //placeholder

            // get the selected dimensions
            if(nonmissing == 0){
                // handle all missing case?
                continue;
            }else {
                meansub = new double[this.G][nonmissing];
                int[] remain = new int[nonmissing];
                int counter1 = 0;
                for (int j = 0; j < this.binary_indices.length; j++) {
                    if (this.data.data[i][this.binary_indices[j]] != -Double.MAX_VALUE) {
                        for (int g = 0; g < this.G; g++) meansub[g][counter1] = this.data.Delta[g][this.binary_indices[j]];
                        remain[counter1] = j;
                        sign[counter1] =  this.data.data[i][binary_indices[j]] > 0 ? 1 : -1;
                        min[counter1] = sign[counter1] > 0 ? 0 : -Double.MAX_VALUE;
                        max[counter1] = sign[counter1] > 0 ? Double.MAX_VALUE : 0;

                        counter1++;
                    }
                }
                covsub = cov.getSubMatrix(remain, remain);
            }
            double[] varj = new double[nonmissing];
            double[][] B = new double[nonmissing][nonmissing];
            varj[0] = covsub.getEntry(0, 0);
            for(int j = 1; j < nonmissing; j++){
                RealMatrix cov_aaj = covsub.getSubMatrix(0, j-1, 0, j-1);
                RealMatrix cov_abj = covsub.getSubMatrix(0, j-1, j, j);
                RealMatrix cov_aajinv = new LUDecomposition(cov_aaj).getSolver().getInverse();

                varj[j] = covsub.getEntry(j, j) - cov_abj.preMultiply(cov_aajinv).preMultiply(cov_abj.transpose())
                        .getEntry(0,0);
                RealMatrix bj = cov_abj.preMultiply(cov_aajinv);
                for(int jj = 0; jj < j; jj++) B[j][jj] = bj.getEntry(jj, 0);
            }
            RealMatrix Bmat = new Array2DRowRealMatrix(B);

            for(int c = 0; c < this.G; c++){
                // if the cause is impossible
                if(this.impossible[i][c]){
                    out[i][c] = 0;
                }else{
                    zerovec[c] = 1;
                    // perform the monte carlo integration
                    RealMatrix V = new Array2DRowRealMatrix(new double[nonmissing][M]);
//                    double[][] V = new double[this.P_binary][M];
//                    for(int jj = 0; jj < this.P_binary; jj++){
//                        V[jj] = U[jj].clone();
//                    }

                    for(int j = 0; j < nonmissing; j++){
                        // update Monte Carlo points
                        double sum = 0;

//                        ArrayList<Double> V2 = new ArrayList<>();
                        V.setRow(j, V.getRowMatrix(j).add(V.preMultiply(Bmat.getRowMatrix(j))).getData()[0]);
                        for(int m = 0; m < M; m++) {
                            V.addToEntry(j, m, meansub[c][j]);
                            if(sign[j] != 0){
                                sum += sign[j] < 0 ? MathUtil.cdf(0, V.getEntry(j, m), Math.sqrt(varj[j])) :
                                        (1-MathUtil.cdf(0, V.getEntry(j, m), Math.sqrt(varj[j])));
                                V.setEntry(j, m, MathUtil.truncNormal(rand, rngN, rngE, V.getEntry(j, m), Math.sqrt
                                        (varj[j]), min[j], max[j],  Double.MAX_VALUE));
                            }
                        }

                        if(sign[j] != 0){
                            // update log prob
                            out[i][c] += Math.log(sum / (M + 0.0));
                        }
                    }
                }
                out[i][c] += Math.log(this.post_prob[c]);
                maxlog = Math.max(out[i][c], maxlog);
            }
            double expsum = 0;
            for(int c = 0; c < this.G; c++){
                if(zerovec[c] != 0) expsum += Math.exp(out[i][c] - maxlog);
            }
            for(int c = 0; c < this.G; c++){
                out[i][c] = zerovec[c] == 0 ? 0 : Math.exp(out[i][c] - maxlog - Math.log(expsum));
            }
            System.out.print(".");

        }


        double sum;
        if(cont_indices.length > 0) {
            for(int i = 0; i < this.N_test; i++){
               if(this.test_case[i]){
                   RealMatrix cont_obs = new Array2DRowRealMatrix(this.data.latent[i]).getSubMatrix(cont_indices, new int[]{0});
                   sum = 0;
                   for(int g = 0; g < this.G; g++){
                       out[i][g] *= MathUtil.multivariate_normal_density(cont_obs, mean_cont[g], cov_bb, NB);
                       sum += out[i][g];
                   }
                   for(int g = 0; g < this.G; g++){
                       out[i][g] /= sum;
                   }
               }
            }
        }

        System.out.println("Finishing resampling membership");
        if(update_prob) this.update_group_prob(out, rand, rngG, same_pop, temp);
        System.out.println("Finishing resampling Fraction Distribution");
        return(out);
    }


    public static void main(String[] args) throws IOException {

        String directory = "/Users/zehangli/";
        String pre = "AAA";
        int N0 = 0;
        int N_test = 800;
        int G = 20;
        int N = N_test + N0;
        int P = 50;
        int nContinuous = 5;
        double miss = 0.5;
        int Nrep = 1;
        String covType = "SSSL";
        String covTypeSim = "Random";
        int Nitr =1000;
        int seed = 321;
        boolean informative = false;
        boolean misspecified = true;
        boolean verbose = true;
        boolean transform = true;
        int rep0 = 0;
        boolean integrate = false;
        boolean NB = false;
        boolean update_with_test = true;
        boolean same_pop = true;
        double sd0 = 0.1;


        if(args.length > 0) {
            int counter = 0;
            N = Integer.parseInt(args[counter]);
            counter++;
            P = Integer.parseInt(args[counter]);
            counter++;
            miss = Double.parseDouble(args[counter]);
            counter++;
            nContinuous = Integer.parseInt(args[counter]);
            counter++;
            Nrep = Integer.parseInt(args[counter]);
            counter++;
            covType = args[counter];
            counter++;
            covTypeSim = args[counter];
            counter++;
            Nitr = Integer.parseInt(args[counter]);
            counter++;
            sd0 = Double.parseDouble(args[counter]);
            counter++;
            seed = Integer.parseInt(args[counter]);
            counter++;
            informative = Boolean.parseBoolean(args[counter]);
            counter++;
            misspecified = Boolean.parseBoolean(args[counter]);
            counter++;
            verbose = Boolean.parseBoolean(args[counter]);
            counter++;
            transform = Boolean.parseBoolean(args[counter]);
            counter++;
            directory = args[counter];
            counter++;
            pre = args[counter];
            counter++;
            G = Integer.parseInt(args[counter]);
            counter++;
            N_test = Integer.parseInt(args[counter]);
            counter++;
            N += N_test;
            rep0 = 0;
            if (args.length > counter) {
                rep0 = Integer.parseInt(args[counter]);
                counter++;
            }
            if (rep0 == 0) {
                Nrep = 1;
            }
            integrate = true;
            if (args.length > counter) {
                integrate = Boolean.parseBoolean(args[counter]);
                counter++;
            }
            NB = false;
            if (args.length > counter) {
                NB = Boolean.parseBoolean(args[counter]);
                counter++;
            }
            update_with_test = true;
            if (args.length > counter) {
                update_with_test = Boolean.parseBoolean(args[counter]);
                counter++;
            }
            same_pop = false;
            if (args.length > counter) {
                same_pop = Boolean.parseBoolean(args[counter]);
                counter++;
            }

        }


        boolean adaptive = true;

        double power = 0;
        double alpha0 = 1;
        boolean update_sparsity = false;
        String expriment_name = pre + covType + "N" + N + "P" + P + "Miss" + miss;
        Random rand = new Random(seed);
        MersenneTwister rngEngin = new MersenneTwister(seed);
        DoubleMersenneTwister rngEngin2 = new DoubleMersenneTwister(seed);

        NormalDistribution rngN = new NormalDistribution(rngEngin, 0, 1);
        Exponential rngE = new Exponential(1, rngEngin2);

        double c = 0.2;
        double multiplier = 2;
        boolean update_group = true;
        if(N_test < 1) update_group = false;
        boolean useVA = false;
        double a_sd0 = 0.0001;
        double b_sd0 = 0.0001;
        double var0 = 1;
        boolean informative_prior = false;
        boolean useDirichlet = true;
        boolean anneal = false;

        for(int rep = (1 + rep0); rep <= (Nrep + rep0); rep ++) {
            seed = seed + rep * 12345;

            String currentdir = directory + expriment_name + "/";
            String currentfile =  expriment_name + "Rep" + rep;

            Latent_classifier model = new Latent_classifier(Nitr, N, N_test, P, G, covType);
            model.Dirichlet = useDirichlet;
            model.data.init_adaptive(sd0, a_sd0, b_sd0, adaptive, power);
            model.update_with_test = update_with_test;
            model.init_prior(G, N_test, alpha0, rand, var0, informative_prior);
            model.savefull = false;

            Simulator simulator = new Simulator(N, P, G);
            simulator.simCov(P, c, multiplier, seed, covTypeSim);
            simulator.set_sim(model, rngEngin, rand, covType, miss, informative, misspecified, nContinuous, transform,
                    alpha0, seed, useVA);


            EvalUtil.savetruth(model, currentdir, currentfile, covType,
                    simulator.prec, simulator.cov, simulator.mean);
            model.keeptruth();
            model.update_prior_prob();


            /** mask some of the data as testing data **/
            for(int i = 0; i < N_test; i++){
                model.test_case[i] = true;
                model.data.membership[i] = G;
            }
            DoubleMersenneTwister rngEngine = new DoubleMersenneTwister(seed);
            Gamma rngG = new Gamma(1.0, 1.0, rngEngine);

            if(covType.equals("PX")){
                model.update_variable_type(model.cov_px);
            }else if(covType.equals("SSSL")){
                model.update_variable_type(model.cov_sssl);
            }

            model.initial_test_label(rand, rngG, rngN, rngE, same_pop);
            model.data.update_groupcounts(model.update_with_test);
            model.findimpossible(); // does nothing without shrink

            /** Model fitting **/
            if(covType.equals("PX")){
                model.cov_px.initial_hotstart(model, 0.1);
                model.setanneal(anneal);
                model.fit_PX_model(seed, verbose, update_group, integrate, NB, same_pop);
                // save results
                EvalUtil.save(model, model.cov_px, currentdir, currentfile, covType, true);
            }else if(covType.equals("SSSL")){
                model.cov_sssl.initial_hotstart(model, 0.1);
                model.setanneal(anneal);
                model.fit_SSSL_model(seed, update_sparsity,  verbose, update_group, integrate, NB, same_pop,
                        currentdir, currentfile);
                if(model.savefull){
                    EvalUtil.save_full(model, model.cov_sssl, currentdir, currentfile, covType, true);
                }
                EvalUtil.save(model, model.cov_sssl, currentdir, currentfile, covType, true);

            }else if(covType.equals("GLASSO")){
//                model.fit_Glasso_model(seed, true);
            }

        }
    }








}
