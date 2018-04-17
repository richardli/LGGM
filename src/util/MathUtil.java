package util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import cern.jet.random.tdouble.Exponential;
import cern.jet.random.tdouble.Gamma;
import math.MultivariateNormal;
import org.apache.commons.math3.distribution.*;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;


public class MathUtil {

    // parse with NA
    public static Integer tryParse(String text) {
        try {
            return Integer.parseInt(text);
        } catch (NumberFormatException e) {
            return Integer.MAX_VALUE * (-1);
        }
    }

    public static double truncStdNormal1D(double x, double mu, Random rand, Exponential rngE){
        if(x == -Double.MAX_VALUE){
            return(rand.nextGaussian());
        }else if(x > 0){
            return(mu + leftTruncStandardNormal(-mu, rand, rngE));
        }else{
            return(mu -leftTruncStandardNormal(-mu, rand, rngE));
        }
    }


    // return pdf(x) = standard Gaussian pdf
    public static double pdf(double x) {
        return Math.exp(-x*x / 2) / Math.sqrt(2 * Math.PI);
    }

    // return pdf(x, mu, signma) = Gaussian pdf with mean mu and stddev sigma
    public static double pdf(double x, double mu, double sigma) {
        return pdf((x - mu) / sigma) / sigma;
    }

    // return cdf(z) = standard Gaussian cdf using Taylor approximation
    public static double cdf(double z) {
        if (z < -8.0) return 0.0;
        if (z >  8.0) return 1.0;
        double sum = 0.0, term = z;
        for (int i = 3; sum + term != sum; i += 2) {
            sum  = sum + term;
            term = term * z * z / i;
        }
        return 0.5 + sum * pdf(z);
    }

    // return cdf(z, mu, sigma) = Gaussian cdf with mean mu and stddev sigma
    public static double cdf(double z, double mu, double sigma) {
        return cdf((z - mu) / sigma);
    }


    // function to extract submatrix by delete index j (square case)
    public static RealMatrix subRemove(RealMatrix mat, int j){
        int[] left = new int[mat.getRowDimension() - 1];
        int counter = 0;
        for(int i = 0; i < left.length + 1; i++){
            if(i == j){continue;}
            left[counter] = i; counter ++;
        }
        RealMatrix sub = mat.getSubMatrix(left, left);
        return(sub);
    }

    // function to extract submatrix by delete row index j
    public static RealMatrix rowRemove(RealMatrix mat, int j){
        int[] left = new int[mat.getRowDimension() - 1];
        int counter = 0;
        for(int i = 0; i < left.length + 1; i++){
            if(i == j){continue;}
            left[counter] = i; counter ++;
        }

        int[] all = new int[mat.getColumnDimension()];
        for(int i = 0; i < all.length; i++) all[i] = i;
        RealMatrix sub = mat.getSubMatrix(left, all);
        return(sub);
    }

    // function to extract submatrix by delete row index j
    public static RealMatrix colRemove(RealMatrix mat, int j){
        int[] left = new int[mat.getColumnDimension() - 1];
        int counter = 0;
        for(int i = 0; i < left.length + 1; i++){
            if(i == j){continue;}
            left[counter] = i; counter ++;
        }

        int[] all = new int[mat.getRowDimension()];
        for(int i = 0; i < all.length; i++) all[i] = i;

        RealMatrix sub = mat.getSubMatrix(all, left);
        return(sub);
    }

    public static double truncGamma(double a, double b,
                                    double min, double max, Random rand){
        if(min > max){
            System.out.printf("Truncation range not right! a = %.4f, b = %.4f, on [%.4f, %.4f]\n", a, b, min, max);
            return(Double.NaN);
        }
        // Gamma in common3 is constructed with rate, scale!!!
        GammaDistribution gamma = new GammaDistribution(a, 1/b);
        if(min < 0) min = 0;
        if(max < 0) max = 0;
        double vmax;
        if(max >= Double.MAX_VALUE){
            vmax = 1;
        }else{
            try {
                vmax = gamma.cumulativeProbability(max);
            }catch(org.apache.commons.math3.exception.ConvergenceException e){
                vmax = 1;
                System.out.printf("Wrong inverse CDF! min = %.4f, max = %.4f\n",
                        min, max);
            }
        }
        double vmin = gamma.cumulativeProbability(min);
        double u = rand.nextDouble() *(vmax - vmin) + vmin;
        if(u < 0 | u > 1){
            System.out.printf("Wrong inverse CDF! min = %.4f, max = %.4f, sample = %.4f\n",
                    vmin, vmax, u);
        }
        if(Math.abs(1-u) < 1E-7){
            return(min);
        }
        double sample = gamma.inverseCumulativeProbability(u);
        if(sample > 100){
            System.out.printf("Large value returned! a = %.4f, b = %.4f, sample = %.4f, on [%.4f, %.4f]\n", a, b,
                    sample, min,
                    max);
        }
        return(sample);
    }
    // function to sample N(0, 1) from (mu, inf), mu > 0
    // Reference: http://arxiv.org/pdf/0907.4010v1.pdf
    public static double leftTruncStandardNormal(double mu, Random rand, Exponential rngE){
        double x = 0;
        double alpha = 0.5 * (mu + Math.sqrt(mu * mu + 4));
        boolean accept = false;
        double max = 100;
        double count = 0;

        while(!accept){
            if(count > max){return(Double.NaN);}
            double z = rngE.nextDouble(alpha) + mu;
            double rho = Math.exp(-1 * (alpha - z) * (alpha - z) / 2.0);
            if(mu > alpha){
                rho *= Math.exp((alpha - mu) * (alpha - mu) / 2.0);
            }
            double u = rand.nextDouble();
            if(u < rho){
                accept = true;
                x = z;
                if(x < mu) System.out.print("?");
            }
            count ++;
        }
        return(x);
    }

    // function to sample tail N(mean, sd) from (mu, inf), mu > mean
    public static double leftTruncNormal(double mean, double sd, double mu, Random rand, Exponential rngE){
        double trunc = (mu - mean) / sd;
        double tildeX = leftTruncStandardNormal(trunc, rand, rngE);
        return(tildeX * sd + mean);
    }

    // function to sample tail N(mean, sd) from (-inf, mu), mu < mean
    public static double rightTruncNormal(double mean, double sd, double mu, Random rand, Exponential rngE){

        double trunc = (mu - mean) / sd;
        double tildeX = leftTruncStandardNormal(trunc * (-1), rand, rngE) * (-1);
        return(tildeX * sd + mean);
    }


    // function to sample truncated normal using inverse CDF
    // cannot handle sampling in far tails
    // pass in standard normal sampler
    public static double truncNormal(Random rand, NormalDistribution rngN,
                                     Exponential rngE,
                                    double mean, double sd,
                                    double min, double max, double boundary){
        // if regular normal
        if(min == Double.MAX_VALUE * (-1) & max == Double.MAX_VALUE){
            return(rngN.sample() * sd + mean);
        }

        // if sampling in left tail
        if(min == Double.MAX_VALUE * (-1) & max < mean){
            return(rightTruncNormal(mean, sd, max, rand, rngE));
        }

        // if sampling in right tail
        if(max == Double.MAX_VALUE  & min > mean) {
            return (leftTruncNormal(mean, sd, min, rand, rngE));
        }

        // if truncated case
        // assign to boundary
        double value;
        double ymin = 0;
        double ymax = 0;

        if(min == boundary * (-1)){
            ymin = rngN.cumulativeProbability(((boundary * (-1) - mean) / sd));
            ymax = rngN.cumulativeProbability(((max - mean) / sd));
            // if cannot tell, e.g., sample N(3, 0.1) on [-100, 0]
            if(Math.abs(ymin - ymax) < 1E-10){
//                return(ymax);
                return(rand.nextDouble() * (max-min) + min);
            }

        }else if (max == boundary){
            ymin = rngN.cumulativeProbability(((min - mean) / sd));
            ymax = rngN.cumulativeProbability(((boundary - mean) / sd));
            // if cannot tell, e.g., sample N(-3, 0.1) on [0, 100]
            if(Math.abs(ymin - ymax) < 1E-10){
//                return(ymin);
                return(rand.nextDouble() * (max-min) + min);
            }

        }


        // sample inverse cumulative probability
        if(ymax < ymin){
            System.out.println("Tolerance too low?");
        }
        double p = rand.nextDouble() * (ymax - ymin) + ymin;
        value = rngN.inverseCumulativeProbability(p) * sd + mean;
//        if(Math.abs(value) > 10) {
//            System.out.printf("lower %.2f, upper %.2f, sampled %.2f, prob %.4f sd %.2f, " +
//                            "mean %.2f\n",
//                    ymin,
//                    ymax,
//                    value, p, sd, mean);
//        }
//        if(value < -1E8){
//            System.out.printf("lower %.6f, upper %.6f, sampled 0\n", ymin, ymax);
//        }
        return(value);
    }

    public static RealMatrix SampleWishart(double df, RealMatrix U, org.apache.commons.math3.random.MersenneTwister
            rand){
        int M = U.getColumnDimension();
        double[][] c = new double[M][M];
        for(int i = 0; i < M; i++){
            // this is not df - i + 1 because java index starts from 0
            // if df = 10, df_chi = 10, 9, 8, ...
            double df_chi = df - i;
            // c[i][i] = Math.sqrt(ChiSquare.random(df_chi, rand));
            c[i][i] =  Math.sqrt(new ChiSquaredDistribution(rand, df_chi).sample());
            for(int j = (i + 1); j < M; j++){
                c[i][j] = rand.nextGaussian();
            }
        }
        RealMatrix CM = new Array2DRowRealMatrix(c);
        RealMatrix CMU = U.preMultiply(CM);
        return (CMU.preMultiply(CMU.transpose()));
    }


	//  function to normalize vectors
	public static double[] norm(double[] x){
		double[] xnorm = new double[x.length];
		double sumx = 0;
		for(int i = 0; i < x.length; i++) sumx += x[i];
		if(sumx == 0){
			for(int i = 0 ; i < xnorm.length; i++) xnorm[i] = 1;
		}else{
			for(int i = 0; i < xnorm.length; i++) xnorm[i] = x[i] / sumx;
		}
		return(xnorm);
	}

	// function to sample inverse beta distribution.	
	public static double truncbeta(BetaDistribution beta, Random rand,
			double min, double max){
		double value = min;
		double ymin = beta.cumulativeProbability(min);
		double ymax = beta.cumulativeProbability(max);
		// handling boundary case
		if(Math.abs(ymax - ymin) < 1e-8){
			//double mean = beta.getNumericalMean();
			return((max + min)/2.0);
			//return((mean < (max + min)/2.0 )? min : max);
		}
		value = beta.inverseCumulativeProbability(rand.nextDouble() * (ymax - ymin) + ymin);
		if(value == 0) 		System.out.printf("lower %.6f, upper %.6f, sampled 0\n", ymin, ymax);
		return(value);
	}

    // function to sample Dirichlet vector using Gamma r.v.
    public static double[] sampleDirichlet(double[] alpha, Gamma rngG){
        double[] x = new double[alpha.length];
        double sum = 0;

        for(int i = 0; i < x.length; i++){
            // sample X ~ Gamma(shape/alpha =  alpha, scale/theta = 1)
            x[i] = rngG.nextDouble(alpha[i], 1);
            sum += x[i];
        }

        for(int i = 0; i < x.length; i++){
            x[i] /= sum;
        }
        return(x);
    }

	// function to find min of selected elements in an array.
	public static double array_min(double[] array, ArrayList<Integer> location){
		double min = Double.MAX_VALUE;
		for(int i : location){
			if(array[i] < min) min = array[i];
		}
		return(min);
	}
	// function to find max of selected elements in an array.
	public static double array_max(double[] array,  ArrayList<Integer> location){
		double max = Double.MAX_VALUE * (-1);
		for(int i : location){
			if(array[i] > max) max = array[i];
		}
		return(max);
	}	
	
	// function to grab certain column of 2d array
	public static double[] grab2(double[][] matrix, int col){
		double[] out = new double[matrix.length];
		for(int i = 0; i < out.length; i++) out[i] = matrix[i][col];
		return(out);
	}
	// function to grab certain column of 2d array
	public static int[] grab2(int[][] matrix, int col){
			int[] out = new int[matrix.length];
			for(int i = 0; i < out.length; i++) out[i] = matrix[i][col];
			return(out);
	}

    // function to get mean from double vector
    public static double getMean(double[] vec){
        double mean = 0;
        for(int i = 0; i < vec.length; i++){
            mean += vec[i];
        }
        mean /= (vec.length + 0.0);
        return(mean);
    }

    // function to get percentile from double vector
    public static double getPercentile(double[] vec, double p){
        Percentile perc = new Percentile();
        return(perc.evaluate(vec, p * 100.0));
    }

//    public static void main(String[] args) {
//        double[] vec = new double[]{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0};
//        System.out.println(getPercentile(vec, 0.1));
//        System.out.println(getPercentile(vec, 0.5));
//        System.out.println(getPercentile(vec, 0.98));
//        System.out.println(getPercentile(vec, 0.23));
//
//    }

    public static double univariate_normal_logdensity(double x, double mu, double sd){
        double ll = -(x - mu) * (x - mu) / 2 / sd / sd;
        ll -= Math.log(sd);
        return(ll);
    }
    public static double univariate_normal_density(double x, double mu, double sd){
        return(Math.exp(univariate_normal_logdensity(x, mu, sd)));
    }

    public static double multivariate_normal_density(RealMatrix x, double[] mu, RealMatrix cov, boolean diag){
        RealMatrix mu_mat = new Array2DRowRealMatrix(mu);
        return(multivariate_normal_density(x, mu_mat, cov, diag));
    }
    public static double multivariate_normal_density(RealMatrix x, RealMatrix mu, RealMatrix cov, boolean diag){
        x = x.subtract(mu);
        return(Math.exp(multivariate_normal_logdensity(x, cov, diag)));
    }

    public static double multivariate_normal_logdensity(double[] x,  double[] mu, RealMatrix cov, boolean diag){
        RealMatrix x_mat = new Array2DRowRealMatrix(x);
        RealMatrix mu_mat = new Array2DRowRealMatrix(mu);
        return(multivariate_normal_logdensity(x_mat, mu_mat, cov, diag));
    }
    public static double multivariate_normal_logdensity(RealMatrix x, RealMatrix mu, RealMatrix cov, boolean diag){
        x = x.subtract(mu);
        return(multivariate_normal_logdensity(x, cov, diag));
    }

    public static double multivariate_normal_logdensity(RealMatrix x, RealMatrix cov, boolean diag){
        if(diag){
            double lik = 0;
            for(int i = 0; i < x.getRowDimension(); i++){
                lik += univariate_normal_logdensity(x.getEntry(i, 0), 0, Math.sqrt(cov.getEntry(i, i)));
            }
            return lik;
        }else{
            LUDecomposition cov_solver = new LUDecomposition(cov);
            RealMatrix inv_cov = cov_solver.getSolver().getInverse();
            double lik = -0.5 * x.preMultiply(inv_cov.preMultiply(x.transpose())).getEntry(0, 0);
            lik -= 0.5 * Math.log(Math.abs(cov_solver.getDeterminant()));
            lik -= x.getRowDimension() * Math.log(Math.PI * 2);
            return lik;
        }
    }


    // more efficient way by computing prec and determinant
    // starndard normal density
    public static double multivariate_normal_logdensity(double[] x){
        double lik = 0;
        for(int i = 0; i < x.length; i++){
            lik -= 0.5 * x[i] * x[i];
        }
        lik -= 0.5 * x.length * Math.log(Math.PI * 2);
        return lik;
    }
    // starndard normal density
    public static double multivariate_normal_logdensity(double[] x, double[] mu){
        double lik = 0;
        for(int i = 0; i < x.length; i++){
            lik -= 0.5 * (x[i] - mu[i]) * (x[i] - mu[i]);
        }
        lik -= 0.5 * x.length * Math.log(Math.PI * 2);
        return lik;
    }

    public static double multivariate_normal_logdensity(double[] x, double[] mu, RealMatrix inv_cov, double det){

        RealMatrix xx = new Array2DRowRealMatrix(x);
        for(int i = 0; i < x.length; i++) xx.addToEntry(i, 0, -mu[i]);
        double lik = -0.5 * xx.preMultiply(inv_cov.preMultiply(xx.transpose())).getEntry(0, 0);
        lik -= 0.5 * Math.log(Math.abs(det));
        lik -= 0.5 * x.length * Math.log(Math.PI * 2);
        return lik;
    }


    // compute CDF for centered multivariate normal distribution
    // default mean to all zero
    public static double multivariate_normal_cdf(double[] lower, double[] upper, RealMatrix cov){
        RealVector meanV = new ArrayRealVector(new double[lower.length]);
        MultivariateNormal mvn = MultivariateNormal.DEFAULT_INSTANCE;
        double value = mvn.cdf(meanV, cov, lower, upper).cdf;
        return(value);
    }

    public static double multivariate_normal_cdf(double[] lower, double[] upper, double[] mean, RealMatrix cov){
        RealVector meanV = new ArrayRealVector(mean);
        MultivariateNormal mvn = MultivariateNormal.DEFAULT_INSTANCE;
        double value = mvn.cdf(meanV, cov, lower, upper).cdf;
        return(value);
    }

    public static double diagCDF( double[] mean, RealMatrix cov, double[] lower, double[] upper){
        NormalDistribution rngN = new NormalDistribution(0, 1);
        double value = 1;
        for(int i = 0; i < mean.length; i++){
            double p0 = 0;
            double p1 = 0;
            if(upper[i] == Double.MAX_VALUE){
                p1 = 1;
            }else{
                p1 = rngN.cumulativeProbability((upper[i] - mean[i]) / Math.sqrt(cov.getEntry(i, i)));
            }
            if(lower[i] == -Double.MAX_VALUE){
                p0 = 0;
            }else{
                p0 = rngN.cumulativeProbability((lower[i] - mean[i]) / Math.sqrt(cov.getEntry(i, i)));
            }
            value *= (p1 - p0);
        }
        return(value);
    }


    public static int discrete_sample(double[] probs, double u){
        int index = probs.length - 1;
        double sum = 0;
        double cumsum = 0;
        for(int i = 0 ; i < probs.length; i++) sum += probs[i];
        for(int i = 0; i < probs.length; i++){
            cumsum += probs[i];
            if(cumsum >= u * sum){
                index = i;
                break;
            }
        }
        return(index);
    }

    public static ArrayList<Integer> discrete_sampleK(int K, double[] probs, Random rand){
        ArrayList<Integer> samp = new ArrayList<>();
        int index = probs.length - 1;
        double[] u = new double[Math.min(K, probs.length)];
        for(int i = 0; i < u.length; i++) u[i] = rand.nextDouble();
        double sum = 0;
        for(int i = 0 ; i < probs.length; i++) sum += probs[i];
        for(int k = 0; k < u.length; k++){
            double cumsum = 0;
            for(int i = 0; i < probs.length; i++){
                cumsum += probs[i];
                if(cumsum >= u[k] * sum){
                    samp.add(i);
                    sum -= probs[i];
                    probs[i] = 0;
                    break;
                }
            }
        }
        return(samp);
    }

}
