package sampler;

import cern.jet.random.tdouble.Exponential;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tdouble.Poisson;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.random.MersenneTwister;
import util.EvalUtil;
import util.MathUtil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by zehangli on 10/26/16.
 */
public class SSSL_COV_model extends COV_model{


    /** hyper priors for SSSL model **/
    int P;
    double[][] zero_P;
    double lambda;
    double v0;
    double v1;
    double[][] alpha;
    double[][] beta;
    double[][] prob;
    public int[][] inclusion;

    // index set for removing one
    int[][] left_index;
    int[][] remove_index;
    List<Integer> onetop;

    public SSSL_COV_model(String type, int P) {
        super(type, P);
    }

    public void sethyper(int G,double v0, double v1, double lambda,
                          double[][] alpha, double[][] beta){
        this.P = alpha.length;
        this.v0 = v0;
        this.v1 = v1;
        this.lambda = lambda;
        this.alpha = new double[P][P];
        this.beta = new double[P][P];
        this.prob = new double[P][P];
        for(int i = 0; i < this.P; i ++){
            for(int j = 0; j < this.P; j++){
                this.alpha[i][j] = alpha[i][j];
                this.beta[i][j] = beta[i][j];
                this.prob[i][j] = alpha[i][j] / (alpha[i][j] + beta[i][j]);

            }
        }


        this.inclusion = new int[P][P];
        for(int i = 0; i < P; i++){
            for(int j = 0; j < P; j++){
                this.inclusion[i][j] = (Math.pow(this.prec.getEntry(i, j), 2) > Math.log(v1 / v0) / (1/2.0/v0/v0 -
                        1/2.0/v1/v1)) ? 1 : 0;
            }
        }

        this.left_index = new int[P][P-1];
        this.remove_index = new int[P][1];
        for(int i = 0; i < P; i++){
            int counter = 0;
            this.remove_index[i][0] = i;
            for(int j = 0; j < P; j++){
                if(i == j) continue;
                this.left_index[i][counter] = j;
                counter ++;
            }
        }
        this.zero_P = new double[G][P];

        this.onetop = new ArrayList<>();
        for (int i = 0; i < this.P; i++) {
            onetop.add(i);
        }
    }


    public void setcov(RealMatrix cov, double lambda){
        for(int i = 0; i < this.P; i++){
            for(int j = 0; j < this.P; j++){
                this.cov.setEntry(i, j, cov.getEntry(i, j));
            }
            this.cov.setEntry(i, i, cov.getEntry(i, i) + lambda);
        }
        this.updateCorrFromCov();
        this.updatePrecFromCov();
    }

    public void resample_SSSL(Mix_data data,
                              Random rand,
                              MersenneTwister rngEngine,
                              NormalDistribution rngN,
                              Exponential rngE,
                              Gamma rngG,
                              boolean update_sparsity,
                              boolean verbose,
                              boolean update_with_test,
                              int[] binary_indices,
                              int[] cont_indices,
                              double[][] penalty_mat) {
        /** resample latent variable within Mix_data **/
        data.resample_Z(rand, rngN, rngE, data.Delta, this.corr_by_tau, verbose);

        /** resample marginal variance for continuous variables **/
        data.resample_tau(binary_indices, cont_indices, this);


        /** resample marginal variance within Mix_data **/
        data.resample_Dfix(rngG, this.invcorr);

        /** expand data **/
        data.expand(data.d, verbose);

        /** resample expanded mean vector **/
        data.resample_expanded(data.sd0, this.prec, this.tau, rngEngine, verbose, update_with_test);

        RealMatrix S = data.computeWprodbyGroup(update_with_test, this.tau, verbose);
        Collections.shuffle(this.onetop);
        for(int j : this.onetop){
            /** resample Precision by SSSL **/
            this.sample_omega(S, data.N + data.G*0, data, rngN, rngG, j, update_with_test, rand, data.power);
            /** resample Graph by element-wise bernoulli **/
            this.sample_prob(j, update_sparsity);
            for(int jj : this.onetop) this.sample_delta(rand, jj, data.d, penalty_mat, false);
        }

        this.updateCorrFromCov();
        this.updateInvCorrFromPrecCov();

        /** recompute D and original mean vector**/
        data.update_Delta(data.d, verbose);

        /** resample sd0 if adaptive **/
        if(data.adaptive){
            data.resample_adaptivesd0(rngG, verbose);
        }

        if(verbose) {
            double[] metrics = EvalUtil.getnorm(this.corr, this.cov_true);
//            double[] metrics = EvalUtil.getnorm(this.invcorr, this.prec_true);

            int[] metrics1 = EvalUtil.getclassification(this.inclusion, this.prec_true);
            System.out.println("Current -------------------------------------");
            System.out.printf("F-norm: %.4f, Infty-norm: %.4f, M-norm %.4f\n",
                    metrics[0], metrics[1], metrics[2]);
            System.out.printf("TP: %d, FP: %d, TN: %d, FN: %d\n",
                    metrics1[0], metrics1[1], metrics1[2], metrics1[3]);
        }

    }

    public double[][] resample_SSSL(
            Latent_model model,
            Mix_data data,
            Random rand,
            MersenneTwister rngEngine,
            NormalDistribution rngN,
            Exponential rngE,
            Gamma rngG,
            boolean update_sparsity,
            boolean verbose,
            boolean integrate,
            boolean NB,
            boolean same_pop,
            int itr,
            int Nitr){
        return(resample_SSSL(model, data, rand, rngEngine, rngN, rngE, rngG, update_sparsity, verbose, integrate, NB,
                same_pop, itr, Nitr, false));
    }

    public double[][] resample_SSSL(
                              Latent_model model,
                              Mix_data data,
                              Random rand,
                              MersenneTwister rngEngine,
                              NormalDistribution rngN,
                              Exponential rngE,
                              Gamma rngG,
                              boolean update_sparsity,
                              boolean verbose,
                              boolean integrate,
                              boolean NB,
                              boolean same_pop,
                              int itr,
                              int Nitr,
                              boolean postselection) {
        double temp = 1;
        if(itr > 0 & itr < Nitr/2 & model.anneal) temp = 0.5 + (itr+1.0)/(Nitr/2.0)* 0.5;
//        double[][] testprob = model.update_group( rand, rngG, rngN, rngE, integrate, NB, same_pop, temp, true);
//        data.resample_Z(rand, rngN, rngE, data.Delta, this.corr_by_tau, verbose);


        /** resample latent variable within Mix_data **/
        double[][] testprob = data.resample_Z_monte(rand, rngN, rngE, data.Delta, this.corr_by_tau, verbose, 50, (Latent_classifier) model, true, rngG, same_pop, temp);

        data.update_groupcounts(model.update_with_test);

        /** resample marginal variance for continuous variables **/
        data.resample_tau(model.binary_indices, model.cont_indices, this);

        /** resample marginal variance within Mix_data **/
        data.resample_Dfix(rngG, this.invcorr);

        /** expand data **/
        data.expand(data.d, verbose);

        /** resample expanded mean vector **/
        data.resample_expanded(data.sd0, this.prec, this.tau, rngEngine, verbose, model.update_with_test);

        if(true) {
            RealMatrix S = data.computeWprodbyGroup(model.update_with_test, tau);
            Collections.shuffle(this.onetop);
            for (int j : this.onetop) {
                /** resample Precision by SSSL **/
                this.sample_omega(S, data.N + data.G * 0, data, rngN, rngG, j, model.update_with_test, rand, data.power);
//                /** resample Graph by element-wise bernoulli **/
                this.sample_prob(j, update_sparsity);
                for (int jj : this.onetop) this.sample_delta(rand, jj, data.d, model.penalty_mat, postselection);
            }
            /** resample Graph by element-wise bernoulli **/
//            for (int jj : this.onetop){
//                this.sample_prob(jj, update_sparsity);
//                this.sample_delta(rand, jj, data.d, model.penalty_mat, postselection);
//            }

            this.updateCorrFromCov();
            this.updateInvCorrFromPrecCov();
        }else{
            this.updateCorrFromCov();
            this.updateInvCorrFromPrecCov();
        }




        /** recompute D and original mean vector**/
        data.update_Delta(data.d, verbose);

        /** resample sd0 if adaptive **/
        if(data.adaptive){
            data.resample_adaptivesd0(rngG, verbose);
        }


        if(verbose) {
            double[] metrics = EvalUtil.getnorm(this.corr, this.cov_true);
//            double[] metrics = EvalUtil.getnorm(this.invcorr, this.prec_true);

            int[] metrics1 = EvalUtil.getclassification(this.inclusion, this.prec_true);
            System.out.println("Current -------------------------------------");
            System.out.printf("F-norm: %.4f, Infty-norm: %.4f, M-norm %.4f\n",
                    metrics[0], metrics[1], metrics[2]);
            System.out.printf("TP: %d, FP: %d, TN: %d, FN: %d\n",
                    metrics1[0], metrics1[1], metrics1[2], metrics1[3]);
        }
        return testprob;
    }


    public void sample_omega(RealMatrix S, int n, Mix_data data, NormalDistribution rngN, Gamma rngG, int i, boolean
            update_with_test, Random rand, double power){

        /** Organization **/
        int N;
        if(update_with_test){
            N = n;
        }else{
            N = n - data.N_test;
        }
        // P-1 x P-1
        RealMatrix Sig11 =
                this.cov.getSubMatrix(this.left_index[i], this.left_index[i]);
        // P-1 x 1
        RealMatrix sig12 =
                this.cov.getSubMatrix(this.left_index[i], this.remove_index[i]);
        RealMatrix s12 =
                S.getSubMatrix(this.left_index[i], this.remove_index[i]);
        double sig22 = this.cov.getEntry(i, i);
        // P-1 x P-1
        //        RealMatrix omega11_inv = this.prec.getSubMatrix(this.left_index[i], this.left_index[i]);
        //        omega11_inv = new LUDecomposition(omega11_inv).getSolver().getInverse();
        RealMatrix omega11_inv = Sig11.subtract(sig12.multiply(sig12.transpose().scalarMultiply(1/sig22)));

        // get C matrix
        RealMatrix Ci =  omega11_inv.scalarMultiply(1.0); // avoid pass by reference...


        /** specify u and v **/
        RealMatrix u_current = this.prec.getSubMatrix(this.left_index[i], this.remove_index[i]);
        RealMatrix v_current = new Array2DRowRealMatrix(new double[1][1]);
        v_current.setEntry(0, 0, 1 / this.cov.getEntry(i, i));
        double s22 = S.getEntry(i, i);

        RealMatrix Dmat = new Array2DRowRealMatrix(new double[P-1][P-1]);
        int counter = 0;
        for(int j = 0; j < P; j++){
            if(j == i){continue;}
            Dmat.setEntry(counter, counter, this.inclusion[i][j] == 0 ? (1.0/v0/v0) : (1.0/v1/v1));
            counter++;
        }

        /** Sample u from ESS **/
         counter = 0;
        for(int j = 0; j < P; j++){
            if(j == i){continue;}
            double add = Dmat.getEntry(counter, counter);
//            add *= (this.cov.getEntry(i, i) * this.cov.getEntry(j, j));
            add *= (this.cov.getEntry(i, i) * data.d[j] * data.d[j]);
            add = (v0 == 0 & this.inclusion[i][j] == 0) ? 0 : add;
            Ci.setEntry(counter, counter, Ci.getEntry(counter, counter) + add / (s22 + 0 + this.lambda / v_current
                    .getEntry(0,0)));
            counter ++;
        }
        Ci = Ci.add(Ci.transpose()).scalarMultiply(0.5);         // make sure it is symmetric
        RealMatrix C = new LUDecomposition(Ci).getSolver().getInverse();
        C = C.add(C.transpose()).scalarMultiply(0.5);
        C = C.scalarMultiply(1 / (s22 + 0 + this.lambda / v_current.getEntry(0,0)));
        RealMatrix mu_u = (s12.preMultiply(C).scalarMultiply(-1));
        //        // double[] off_diag =
        //        //          new MultivariateNormalDistribution(mu_post, C.getData()).sample();
        //        // using Chol decomposition directly,
        //        // Sigma = LL^T, z ~ N(0, I) then (mu + Lz) gives N(mu, Sigma)
        //        RealMatrix L = new CholeskyDecomposition(C).getL();
        //        RealMatrix z = new Array2DRowRealMatrix(rngN.sample(P - 1));
        //        u = (z.preMultiply(L)).add(mu_u).getColumn(0);
        double[][] zero = new double[1][1];
        RealMatrix zeromat = new Array2DRowRealMatrix(zero);
        RealMatrix d_minus_j = new Array2DRowRealMatrix(new double[P-1][P-1]);
        counter = 0;
        for(int j = 0; j < P; j++) {
            if (j != i){
                d_minus_j.setEntry(counter, counter, data.d[j]*data.d[j]);
                counter++;
            }
        }

        double[] u = ESSsampler.sample(u_current, 1, mu_u, C, false, v_current.getColumn(0), d_minus_j, omega11_inv,
                        Dmat, zeromat, rand);


        /** Sample sqrt(v) from ESS **/
//        v_current.setEntry(0, 0, Math.sqrt(v_current.getEntry(0, 0)));
//
//        double[][] sd2 = new double[1][1];
//        sd2[0][0] =  1 / (s22 + 1);
//        RealMatrix sd2mat = new Array2DRowRealMatrix(sd2);
//        double[] n_and_lambda = new double[2];
//        n_and_lambda[0] = N;
//        n_and_lambda[1] = this.lambda;
//        RealMatrix umat = new Array2DRowRealMatrix(u);
//        RealMatrix cross = omega11_inv.preMultiply(umat.transpose()).preMultiply(umat).preMultiply(omega11_inv);
//        double[] v_sqrt = ESSsampler.sample(v_current, 2, zeromat, sd2mat, true, n_and_lambda, omega11_inv, cross,
//                umat, Dmat, rand);
//        double vv = v_sqrt[0]*v_sqrt[0];

         /** Sample v from ESS using normal approximation **/

        double[][] sd2 = new double[1][1];
        sd2[0][0] = Math.pow(1 / (s22 + 1), 2) * (N / 2.0 + 1); //-power
        RealMatrix sd2mat = new Array2DRowRealMatrix(sd2);
        double[][] mu = new double[1][1];
        mu[0][0] = 1 / (s22 + 1) * (N / 2.0 + 1 ); //-power
        RealMatrix mumat = new Array2DRowRealMatrix(mu);
        double[] n_and_lambda = new double[2];
        n_and_lambda[0] = N;
        n_and_lambda[1] = this.lambda;
        RealMatrix umat = new Array2DRowRealMatrix(u);
        RealMatrix cross = omega11_inv.preMultiply(umat.transpose()).preMultiply(umat).preMultiply(omega11_inv);
        double[] v = ESSsampler.sample(v_current, 3, mumat, sd2mat, true, n_and_lambda, omega11_inv, cross,
                umat, Dmat, rand);
        double vv = Math.abs(v[0]);




//        vv = rngG.nextDouble(1 + N/2.0 + power,  (1 + this.lambda * data.d[i] * data.d[i]+ S.getEntry(i,i)) / 2.0);


        /** Transfer back to omega **/
        double addterm = (umat.preMultiply(omega11_inv).preMultiply(umat.transpose())).getEntry(0, 0);
        counter = 0;

        // ***
        for(int j = 0; j < P; j++){
            if(j == i){
                this.prec.setEntry(i, j, vv + addterm);
            }else{
                this.prec.setEntry(i, j, u[counter]);
                this.prec.setEntry(j, i, u[counter]);
                counter++;
            }
        }
//        this.updateCovFromPrec();
        RealMatrix newSig11 = omega11_inv.add(cross.scalarMultiply(1/vv));
        newSig11 = newSig11.add(newSig11.transpose()).scalarMultiply(0.5);
        RealMatrix newSig12 = umat.preMultiply(omega11_inv).scalarMultiply(-1/vv);
        counter = 0;
        for(int j = 0; j < P; j++){
            if(j == i){
                this.cov.setEntry(i, j, 1/vv);
            }else{
                this.cov.setEntry(i, j, newSig12.getEntry(counter, 0));
                this.cov.setEntry(j, i, newSig12.getEntry(counter, 0));
                counter++;
            }
        }
        int counter1 = 0;
        for(int j = 0; j < P; j++){
            int counter2 = 0;
            for(int k = 0; k < P; k++){
                if(j != i & k!= i){
                    this.cov.setEntry(j, k, newSig11.getEntry(counter1, counter2));
                    counter2++;
                }
            }
            if(j!=i) counter1++;
        }
    }

    /** Sample the j-th column of the prob matrix, and update the transpose **/
    public void sample_prob(int j, boolean update_sparsity){
        if(!update_sparsity){
            return;
        }
        for(int i = 0; i < this.P; i++){
            if(i == j) continue;
            BetaDistribution rngB = new BetaDistribution(this.alpha[i][j] + this.inclusion[i][j], this.beta[i][j] + 1 - this.inclusion[i][j]);
            this.prob[i][j] = rngB.sample();
            this.prob[j][i] = this.prob[i][j];
        }
    }

    /** Sample the j-th column of the delta matrix, and update the transpose **/
    public void sample_delta(Random rand, int j, double[] d, double[][] penalty, boolean skip_selection){
        if(skip_selection){
            return;
        }
        for(int i = 0; i < this.P; i++){
            if(i == j) continue;
            if(penalty[i][j] != 0){
                this.inclusion[i][j] = 1;
                this.inclusion[j][i] = 1;
                continue;
            }
            double p = this.prob[i][j];
            double dd = Math.sqrt(this.cov.getEntry(j, j)) * Math.sqrt(this.cov.getEntry(i, i));
            double w0 = MathUtil.univariate_normal_logdensity(this.prec.getEntry(i, j), 0, v0/dd) + Math.log(1 -
                    p);
            double w1 = MathUtil.univariate_normal_logdensity(this.prec.getEntry(i, j), 0, v1/dd) + Math.log(p);
            double wm = Math.max(w0, w1);
            double w = Math.exp(w1 - wm) / (Math.exp(w1 - wm) + Math.exp(w0 - wm));

            this.inclusion[i][j] = rand.nextDouble() < w ? 1:0;
            this.inclusion[j][i] = this.inclusion[i][j];
        }
    }


}
