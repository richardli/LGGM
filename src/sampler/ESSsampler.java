package sampler;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

/**
 * Created by zehangli on 1/18/18.
 */
public class ESSsampler {

//
//    public static double[] sample (double[] current, int type, double[] mu, double[][] sigma, boolean diagonal,
//                                   double[][] par, Random rand){
//        return(sample(current, type, mu, sigma, diagonal, par, rand, 100));
//    }
    public static double[] sample (RealMatrix current, int type, RealMatrix mu, RealMatrix sigma, boolean diagonal,
                                   double[] par0, RealMatrix par1, RealMatrix par2, RealMatrix par3, RealMatrix par4,
                                   Random rand){
        return(sample(current, type, mu, sigma, diagonal, par0, par1, par2, par3, par4, rand, 100));
    }
    /**
     * THis version only used for sampling CSMF
     * @param current current vector
     * @param type
     * @param rand
     * @param maxitr
     * @return
     */
    public static double[] sample (double[] current, int type, double[] mu, double[] var, double[] par, Random rand, int
            maxitr, boolean fixfirstzero){

        // initiate new samples
        int p = current.length;
        int start = 0;
        if(fixfirstzero){
            mu[0] = 0;
            start = 1;
        }

        double[] newsample = new double[p];
        double[] prior = new double[p];
        for(int i = start; i < p; i++){
            current[i] = current[i] - mu[i];
        }

        // sample from prior
        for(int i = start; i < p; i++) prior[i] = rand.nextGaussian() * Math.sqrt(var[i]);

        // set parameters
        double u = rand.nextDouble();
        double logy0 = Math.log(u) + evalloglik(type, current, mu, par);
        double logy;
        double theta = rand.nextDouble() * 2 * Math.PI;
        double theta1 = theta - 2 * Math.PI;
        double theta2 = theta;

        // slice sampling
        int itr = 0;
        while(true){
            for(int i = 0; i < p; i++) newsample[i] =  current[i] * Math.cos(theta) + prior[i] * Math.sin(theta);
            logy = evalloglik(type, newsample, mu, par);
            if(logy > logy0){
                break;
            }else{
                if(theta < 0){
                    theta1 = theta;
                }else{
                    theta2 = theta;
                }
                theta = rand.nextDouble() * (theta2 - theta1) + theta1;
            }
            itr ++;
            if(itr > maxitr){
                System.out.println("Max iteration reached in ESS!");
                break;
            }
        }

        // return output
        for(int i = start; i < p; i++){
            newsample[i] = newsample[i] + mu[i];
        }
        return(newsample);
    }
    /**
     *
     * @param current current vector: p * 1
     * @param type
     * @param rand
     * @param maxitr
     * @return
     */
    public static double[] sample (double[] current, int type, double[] mu, double[][] sigma, boolean diagonal,
                                   double[] par0, RealMatrix par1, RealMatrix par2, RealMatrix par3, RealMatrix par4,
                                   Random rand, int maxitr){
        return(sample(new Array2DRowRealMatrix(current), type, new Array2DRowRealMatrix(mu), new Array2DRowRealMatrix
                (sigma), diagonal, par0, par1, par2, par3, par4, rand, maxitr));
    }

    public static double[] sample (RealMatrix current, int type, RealMatrix mu, RealMatrix sigma, boolean diagonal,
                                   double[] par0, RealMatrix par1, RealMatrix par2, RealMatrix par3, RealMatrix par4,
                                   Random rand, int maxitr){
        double thres = 1E-6; // minimum difference

        // initiate new samples
        int p = current.getRowDimension();
        RealMatrix newsample;
        RealMatrix prior = new Array2DRowRealMatrix(new double[p]);
        for(int i = 0; i < p; i++){
            current.setEntry(i, 0, current.getEntry(i, 0) - mu.getEntry(i, 0));
        }

        // sample from prior
        if(diagonal){
            for(int i = 0; i < p; i++) prior.setEntry(i, 0, rand.nextGaussian() * Math.sqrt(sigma.getEntry(i, i)));
        }else{
            RealMatrix L = new CholeskyDecomposition(sigma, 1E-10, 1E-20).getL();
            RealMatrix z = new Array2DRowRealMatrix(new double[p][1]);
            for(int i = 0; i < p; i++) z.setEntry(i, 0, rand.nextGaussian());
            prior = z.preMultiply(L);
        }

        // set parameters
        double u = rand.nextDouble();
        double logy0 = Math.log(u) + evalloglik(type, current, mu, par0, par1, par2, par3, par4);
        double logy;
        double theta = rand.nextDouble() * 2 * Math.PI;
        double theta1 = theta - 2 * Math.PI;
        double theta2 = theta;

        // slice sampling
        int itr = 0;
        while(true){
            newsample = current.scalarMultiply(Math.cos(theta));
            newsample = newsample.add(prior.scalarMultiply(Math.sin(theta)));
            logy = evalloglik(type, newsample, mu, par0, par1, par2, par3, par4);

            if(logy > logy0){
                break;
            }else{
                if(theta < 0){
                    theta1 = theta;
                }else{
                    theta2 = theta;
                }
                theta = rand.nextDouble() * (theta2 - theta1) + theta1;
            }
            itr ++;
            if(itr > maxitr) {
                System.out.println("Max iteration reached in ESS!");
                break;
            }
            double diff = 0;
            for(int i = 0; i < current.getRowDimension(); i++) diff = Math.max(diff, Math.abs(current.getEntry(i, 0)
                    - newsample.getEntry(i, 0)));
            if(diff < thres){
                break;
            }
        }

        // return output
        newsample = newsample.add(mu);
        return(newsample.getColumn(0));
    }



    public static double evalloglik(int type, double[] sample, double[] mu, double[] par){
        double loglik = 0;
        if(type == 5){
            double[] prob = new double[par.length];
//            prob[0] = 1;
            double sum = 0;
            for(int i = 0; i < prob.length; i++){
                prob[i] = Math.exp(sample[i] + mu[i]);
                sum += prob[i];
            }
            for(int i = 0; i < prob.length; i++) prob[i] /= sum;
            for(int i = 0; i < par.length; i++) loglik += (par[i] + 0.0) * Math.log(prob[i]);

        }
        return(loglik);
    }



    public static double evalloglik(int type, RealMatrix sample, RealMatrix mu,  double[] par0, RealMatrix par1,
                                    RealMatrix par2, RealMatrix par3 , RealMatrix par4){
        double loglik = 0;

        sample = sample.add(mu);

        if(type == 1){
            /** sampling u | v, P-1 dimensional, for omega matrix **/
            /**
             * par0 = c(v)
             * par1 = (P-1) * (P-1) matrix  of  Sigma_{-j, -j} (current)
             * par2 = (P-1) * (P-1) matrix  of  Omega_{-j, -j}^(-1)
             * par3 = (P-1) * (P-1) matrix  of  diag(1 / v)
             */
            // cross product
            double phi;
            RealMatrix tmp = sample.preMultiply(par2);
            // new Sigma_11
            par2 = par2.add(tmp.transpose().preMultiply(tmp).scalarMultiply(1/par0[0]));
            // D_star
            for(int i = 0; i < par3.getColumnDimension(); i++){
                phi = (par2.getEntry(i, i)  -  par1.getEntry(i,i) ) / (2 * par0[0]);
                par3.setEntry(i, i, par3.getEntry(i, i) * phi);
            }
            loglik = (-1) * sample.preMultiply(par3).preMultiply(sample.transpose()).getEntry(0,0);

            // add last term
            for(int i = 0; i < par2.getColumnDimension() ; i++) loglik -= 0.5 / (par2.getEntry(i,i));


        }else if(type == 2){
            /** sampling v | u, 1 dimensional, for omega matrix
             *  par0 = c(n, lambda)
             *  par1 = Omega_{-j, -j}^(-1)        : P-1 * P-1
             *  par2 = cross_product              : P-1 * P-1
             *  par3 = u                          : P-1 * 1
             *  par4 = 1/v_{jk}                   : P-1 * P-1
             **/
            if(sample.getEntry(0, 0) < 0) return(-Double.MAX_VALUE);
            par1 = par1.scalarMultiply(par0[1]); // lambda * Omega_{-j, -j} ^ (-1)
            // new Sigma_{-j, -j}
            par2 = par1.add(par2.scalarMultiply(1 / sample.getEntry(0, 0) / sample.getEntry(0, 0)));
            for(int i = 0; i < par1.getColumnDimension(); i++){
                par1.setEntry(i, i, par2.getEntry(i, i) * par4.getEntry(i, i));
            }
            loglik = (-1) * par3.preMultiply(par1).preMultiply(par3.transpose()).getEntry(0, 0);
            loglik /= 2 * Math.pow(sample.getEntry(0, 0), 2) ;
            loglik += (par0[0] - 1) * Math.log(sample.getEntry(0, 0));

            // add last term
            for(int i = 0; i < par2.getColumnDimension() ; i++) loglik -= 0.5 / (par2.getEntry(i,i));


        }else if(type == 3){
            /** sampling v | u, 1 dimensional, for omega matrix, using Gaussian approximation
             *  par0 = c(n, lambda)
             *  par1 = Omega_{-j, -j}^(-1)        : P-1 * P-1
             *  par2 = cross_product              : P-1 * P-1
             *  par3 = u                          : P-1 * 1
             *  par4 = 1/v_{jk}                   : P-1 * P-1
             **/
            par1 = par1.scalarMultiply(par0[1]); // lambda * Omega_{-j, -j} ^ (-1)
            // new Sigma_{-j, -j}
            par2 = par1.add(par2.scalarMultiply(1 / sample.getEntry(0, 0) / sample.getEntry(0, 0)));
            for(int i = 0; i < par1.getColumnDimension(); i++){
                par1.setEntry(i, i, par2.getEntry(i, i) * par4.getEntry(i, i));
            }
            loglik = (-1) * par3.preMultiply(par1).preMultiply(par3.transpose()).getEntry(0, 0);
            loglik /= 2 * Math.abs(sample.getEntry(0, 0));
        }else if(type == 4){
            /** sampling 1/tau  **/
            /**
             * par0 = c(N-2)
             */
            loglik = par0[0] * Math.log(Math.abs(sample.getEntry(0,0)));
            if(sample.getEntry(0,0) <= 0) loglik = - Double.MAX_VALUE;
        }else{
            System.out.println("Wrong log likelihood type!");
        }

        return(loglik);
    }

}
