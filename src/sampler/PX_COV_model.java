package sampler;

import cern.jet.random.tdouble.Exponential;
import cern.jet.random.tdouble.Gamma;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.MersenneTwister;
import util.EvalUtil;
import util.MathUtil;

import java.util.HashMap;
import java.util.Random;

/**
 * Created by zehangli on 10/26/16.
 */
public class PX_COV_model extends COV_model {


//    /** hyper priors for PX model **/
//    double sigma_px;

    public PX_COV_model(String type, int P) {
        super(type, P);
    }

//    /** set hyper parameter
//     *
//     * @param sigma_px sd of delta
//     */
//    public void sethyper(double sigma_px){
//        this.sigma_px = sigma_px;
//    }


    /**
     * Sampling covariance and precision matrix
     *
     * @param n number of samples in the data
     * @param Sigma0 Mean of the extended covariance matrix (Sigma0 = I_p + W)
     * @param rand Random generator
     * @param verbose whether to print diagnostics messages
     */
    public void resample_PX(int n, int N_test, RealMatrix Sigma0, MersenneTwister rand, boolean verbose, boolean
            update_with_test){

        RealMatrix invSigma0 = new LUDecomposition(Sigma0).getSolver().getInverse();
        invSigma0 = invSigma0.add(invSigma0.transpose());
        invSigma0 = invSigma0.scalarMultiply(0.5);

        // // old implementation of Wishart, seems not correct...
        // RealMatrix lower = new CholeskyDecomposition(invSigma0).getL();
        // double[][] precNew = rngW.random(N + this.P + 1.0, lower.getData(), rand);

        RealMatrix upper = new CholeskyDecomposition(invSigma0).getLT();
        if(update_with_test){
            double[][] precNew = MathUtil.SampleWishart(n + this.P + 1.0, upper, rand).getData();
            this.prec = new Array2DRowRealMatrix(precNew);
        }else{
            double[][] precNew = MathUtil.SampleWishart(n - N_test + this.P + 1.0, upper, rand).getData();
            this.prec = new Array2DRowRealMatrix(precNew);
        }

        this.updateAllFromPrec();

        if(verbose){
            this.print_sim_message();
        }
    }

    /**
     * Sample one iteration of PX model as a whole
     *
     * Key items for update at each iteration
     *  1. latent variable Z (data.latent)
     *  2. expanded latent variable W (data.expanded)
     *  3. latent mean Delta (data.Delta)
     *  4. expanded latent mean gamma (data.Delta_expanded)
     *  5. marginal standard error d
     *  6. cov
     *  7. corr
     *  8. prec
     *  9. invCorr
     */
    public void resample(Mix_data data,
                         Random rand,
                         MersenneTwister rngEngine,
                         NormalDistribution rngN,
                         Exponential rngE,
                         Gamma rngG,
                         boolean verbose,
                         boolean update_with_test,
                         int[] binary_indices,
                         int[] cont_indices){

            data.update_groupcounts(update_with_test);

            /** resample latent variable within Mix_data [1] **/
            data.resample_Z(rand, rngN, rngE, data.Delta,
                    this.corr_by_tau, verbose);

            /** resample marginal variance for continuous variables **/
            data.resample_tau(binary_indices, cont_indices, this);


            /** resample marginal variance within Mix_data [5] **/
            data.resample_D(rngG, this.invcorr);
            //data.resample_D_carryover(this.cov);

            /** Update expanded mean and data within Mix_data [2][4] **/
            data.expand(data.d, verbose);

             /** resample expanded mean vector [3]**/
             data.resample_expanded(data.sd0, this.prec, this.tau, rngEngine, verbose, update_with_test);

            /** resample cov matrix, and update prec and corr matrix [6][7][8][9] **/
            this.resample_PX(data.N + data.G*0, data.N_test, data.computeWprodPlusIp(verbose,
                    update_with_test, this.tau),
                rngEngine, verbose, update_with_test);

//            /** resample expanded mean vector [3]**/
//            data.resample_expanded_conj(data.sd0, this.cov, rngEngine, verbose, update_with_test);

            /** recompute D and original mean vector [5][1]**/
             data.update_Delta(data.d, verbose);

            /** resample sd0 if adaptive **/
            if(data.adaptive){
                data.resample_adaptivesd0(rngG, verbose);
            }


            /**  print message **/
            if(verbose) {
                double[] metrics = EvalUtil.getnorm(this.corr, this.cov_true);
                System.out.println("Current -------------------------------------");
                System.out.printf("F-norm: %.4f, Infty-norm: %.4f, M-norm %.4f\n",
                        metrics[0], metrics[1], metrics[2]);
            }
    }

    public double[][] resample(
                         Latent_model model,
                         Mix_data data,
                         Random rand,
                         MersenneTwister rngEngine,
                         NormalDistribution rngN,
                         Exponential rngE,
                         Gamma rngG,
                         boolean verbose,
                         boolean integrate,
                         boolean NB,
                         boolean same_pop,
                         int itr,
                         int Nitr){

        /** resample latent variable within Mix_data [1] **/
        data.resample_Z(rand, rngN, rngE, data.Delta,
                this.corr_by_tau, verbose);

        /** resample marginal variance for continuous variables **/
//        this.resample_tau();

        double temp = 1;
        if(itr < Nitr/2 & model.anneal) temp = 0.5 + Math.sqrt(itr+1.0)/Math.sqrt(Nitr/2.0)* 0.5;
        double[][] testprob = model.update_group(rand, rngG, rngN, rngE, integrate, NB, same_pop, temp, true);

        data.update_groupcounts(model.update_with_test);

        /** resample marginal variance for continuous variables **/
        data.resample_tau(model.binary_indices, model.cont_indices, this);

        /** resample marginal variance within Mix_data [5] **/
        data.resample_D(rngG, this.invcorr);
        //data.resample_D_carryover(this.cov);

        /** Update expanded mean and data within Mix_data [2][4] **/
        data.expand(data.d, verbose);

        /** resample expanded mean vector [3]**/
        data.resample_expanded(data.sd0, this.prec, this.tau, rngEngine, verbose, model.update_with_test);

        /** resample cov matrix, and update prec and corr matrix [6][7][8][9] **/
        this.resample_PX(data.N + data.G*0, data.N_test, data.computeWprodPlusIp(verbose, model.update_with_test, this
                .tau),
                rngEngine, verbose, model.update_with_test);


        /** recompute D and original mean vector [5][1]**/
        data.update_Delta(data.d, verbose);

        /** resample sd0 if adaptive **/
        if(data.adaptive){
            data.resample_adaptivesd0(rngG, verbose);
        }


        /**  print message **/
        if(verbose) {
            double[] metrics = EvalUtil.getnorm(this.corr, this.cov_true);
            System.out.println("Current -------------------------------------");
            System.out.printf("F-norm: %.4f, Infty-norm: %.4f, M-norm %.4f\n",
                    metrics[0], metrics[1], metrics[2]);
        }
        return(testprob);
    }


}
