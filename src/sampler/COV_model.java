package sampler;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.*;

/**
 * Created by zehangli on 5/21/16.
 */
public class COV_model {

    public String type;            // String of the covariance type
    public int P;                  // Dimension of the covariance matrix

    // Four matrices of interest
    public RealMatrix cov;
    public RealMatrix corr;
    public RealMatrix prec;
    public RealMatrix invcorr;

    public double[] tau;
    public RealMatrix corr_by_tau;


    // Average estimator of corr and inverse corr matrix
    public RealMatrix corr_ave;
    double corr_add_count = 0;
    public RealMatrix invcorr_ave;
    double prec_add_count = 0;
    public int[][] inclusion_ave;

    public RealMatrix cov_true;    // Truth (for simulation)
    public RealMatrix prec_true;   // Truth (for simulation)


    /** initialize Covariance model to I_p
     *
     * @param type string of the covariance estimator type
     *             1. PX
     *             2. SSSL
     *             3. Glasso
     * @param P dimension of the covariance matrix
     */
    public COV_model(String type, int P){
        this.type = type;
        this.tau = new double[P];
        for(int i = 0; i < P; i++) this.tau[i] = 1;
        this.P = P;
        double[][] IpVector = new double[P][P];
        // initialize average to 0
        this.corr_ave = new Array2DRowRealMatrix(IpVector);
        this.invcorr_ave = new Array2DRowRealMatrix(IpVector);
        this.inclusion_ave = new int[P][P];

        for(int i = 0; i < this.P; i++){
            IpVector[i][i] = 1;
        }
        this.cov = new Array2DRowRealMatrix(IpVector);
        this.corr = new Array2DRowRealMatrix(IpVector);
        this.corr_by_tau = new Array2DRowRealMatrix(IpVector);
        this.prec = new Array2DRowRealMatrix(IpVector);
        this.invcorr = new Array2DRowRealMatrix(IpVector);
        this.cov_true = new Array2DRowRealMatrix(IpVector);
        this.prec_true = new Array2DRowRealMatrix(IpVector);

    }

    public void initial_hotstart(double[][] R){
        for(int i = 0 ; i < this.P; i++) {
            for (int j = 0; j < this.P; j++) {
                this.cov.setEntry(i, j, R[i][j]);
                this.corr.setEntry(i, j, R[i][j]);
                this.corr_by_tau.setEntry(i, j, R[i][j]);
            }
        }
        this.invcorr = new LUDecomposition(this.corr).getSolver().getInverse();
//       this.updatePrecFromCov();
//       this.updateAllFromPrec();
    }

    public void initial_hotstart(Latent_model model, double add){
        this.cov = this.cov.scalarMultiply(add);
        int nn = model.N;
        if(model.N > model.N_test) nn = model.N - model.N_test;

        double[][] latents = new double[nn][P];

        int index = 0;
        for(int i = 0; i < model.N; i++){
            int gtmp = model.data.membership[i];
            if(model.data.membership[i] == model.data.G ){
                if(model.N > model.N_test){
                    continue;
                }else{
                    gtmp = model.data.membership_test[i];
                }
            }
            for(int j = 0; j < P; j++){
                latents[index][j] = model.data.latent[index][j] - model.data.Delta[gtmp][j];
            }
            index++;
        }
        RealMatrix crossprod = new Array2DRowRealMatrix(latents);
        crossprod = crossprod.preMultiply(crossprod.transpose()).scalarMultiply(1/(nn+0.0));
//        for(int j = 0; j < this.P; j++){
//            for(int jj = 0; jj < this.P; jj++){
//                if(j != jj) crossprod.setEntry(j, jj, 0);
//            }
//        }
//        RealMatrix Dinv = new Array2DRowRealMatrix(new double[P][P]);
//        for(int j = 0; j < this.P; j++){
//            double tmp = crossprod.getEntry(j, j) == 0 ? 1 : 1/Math.sqrt(crossprod.getEntry(j, j));
//            Dinv.setEntry(j, j, tmp);
//        }
//        crossprod = Dinv.preMultiply(crossprod).preMultiply(Dinv);
        this.cov = this.cov.add(crossprod);
        this.prec = new LUDecomposition(this.cov).getSolver().getInverse();
        updateCovFromPrec();
        updateCorrFromCov();
        updateInvCorrFromCorr();
        print_sim_message();
        System.out.println("Initialize to empirical correlation matrix");
    }



    /**
     * Read/Set true covariance matrix from the file
     *
     * @param file String of file address
     * @throws IOException
     */
    public void readSigma(String file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        int counter = 0;
        while ((line = br.readLine()) != null) {
            String[] temp = line.split(",");
            for(int j = 0; j < this.P; j++){
                this.cov_true.setEntry(counter, j, Double.parseDouble(temp[j]));
            }
            counter ++;
        }
        br.close();
    }

    /**
     * Read/Set true precision matrix from file
     *
     * @param file String of file address
     * @throws IOException
     */

    public void readOmega(String file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        int counter = 0;
        while ((line = br.readLine()) != null) {
            String[] temp = line.split(",");
            for(int j = 0; j < this.P; j++){
                this.prec_true.setEntry(counter, j, Double.parseDouble(temp[j]));
            }
            counter ++;
        }
        br.close();
    }

    public void setOmega(double[][] prec)  {
        for(int i = 0 ; i < this.P; i++) {
            for (int j = 0; j < this.P; j++) {
                this.prec_true.setEntry(i, j, prec[i][j]);
            }
        }
    }

    public void setSigma(double[][] cov)  {
        for(int i = 0 ; i < this.P; i++) {
            for (int j = 0; j < this.P; j++) {
                this.cov_true.setEntry(i, j, cov[i][j]);
            }
        }
    }
    public void setTruth(double[][] cov, double[][] prec){
        this.setOmega(prec);
        this.setSigma(cov);
    }


    /**
     * Calculate Corr matrix from covariance matrix
     */
    public void updateCorrFromCov(){
        for(int j = 0; j < this.P; j++){
            for(int jj = 0; jj < this.P; jj++){
                double temp = this.cov.getEntry(j, j) * this.cov.getEntry(jj, jj);
                this.corr.setEntry(j, jj, this.cov.getEntry(j, jj) / Math.sqrt(temp));
                this.corr_by_tau.setEntry(j, jj, this.corr.getEntry(j, jj) * this.tau[j] * this.tau[jj]);
            }
        }
    }
    public void updateInvCorrFromPrecCov() {
        for (int j = 0; j < this.P; j++) {
            for (int jj = 0; jj < this.P; jj++) {
                double temp = this.cov.getEntry(j, j) * this.cov.getEntry(jj, jj);
                this.invcorr.setEntry(j, jj, this.prec.getEntry(j, jj) * Math.sqrt(temp));
            }
        }
        print_sim_message();
    }

    /**
     * Calculate Cov matrix from Precision matrix
     */
    public void updateCovFromPrec(){
        this.cov = new LUDecomposition(this.prec).getSolver().getInverse();
    }

    /**
     * Calculate Precision matrix from Covariance matrix
     */
    public void updatePrecFromCov(){
        this.prec = new LUDecomposition(this.cov).getSolver().getInverse();
    }

    /**
     * Calculate InvCorrelation matrix from Corre matrix
     */
    public void updateInvCorrFromCorr(){
        this.invcorr = new LUDecomposition(this.corr).getSolver().getInverse();
    }

    /**
     * Calculate InvCorrelation matrix from Corre matrix
     */
    public void updateAllFromPrec(){
        this.cov = new LUDecomposition(this.prec).getSolver().getInverse();
        double [][] Dinv = new double[this.P][this.P];
        double [][] D = new double[this.P][this.P];
        double [][] Lambda = new double[this.P][this.P];
        for(int i = 0; i < this.P; i++){
            D[i][i] = Math.sqrt(this.cov.getEntry(i, i));
            Dinv[i][i] = 1 / D[i][i];
            Lambda[i][i] = this.tau[i];
        }
        RealMatrix Dinvmat = new Array2DRowRealMatrix(Dinv);
        RealMatrix Dmat = new Array2DRowRealMatrix(D);
        RealMatrix Lambdamat = new Array2DRowRealMatrix(Lambda);
        this.invcorr = Dmat.preMultiply(this.prec.preMultiply(Dmat));
        this.corr = Dinvmat.preMultiply(this.cov.preMultiply(Dinvmat));
        this.corr_by_tau = Lambdamat.preMultiply(this.corr.preMultiply(Lambdamat));

        double tmp_max = 0;
        for(int i = 0; i < this.P; i++) tmp_max = Math.max(tmp_max, this.invcorr.getEntry(i, i));
        if(tmp_max > 10){
            System.out.println(".");
        }
        System.out.printf("diagonal R inverse           : max: %.4f\n", tmp_max);
        print_sim_message();

    }

    /**
     * Calculate average Correlation matrix
     */
    public void updateCorrAve(){
        this.corr_ave = this.corr_ave.scalarMultiply((this.corr_add_count + 0.0));
        this.corr_ave = this.corr_ave.add(this.corr);
        this.corr_add_count ++;
        this.corr_ave = this.corr_ave.scalarMultiply(1.0 / (this.corr_add_count + 0.0));
    }

    /**
     * Calculate average Inclusion
     */
    public void updateInclusionAve(int[][] inclusion){
        for(int i = 0; i < this.P; i++){
            for(int j = 0; j < this.P; j++){
                this.inclusion_ave[i][j] += inclusion[i][j];
            }
        }
    }

    /**
     *  Calculate average prec matrix
     */
    public void updateInvCorrAve(){
        this.invcorr_ave = new LUDecomposition(this.corr_ave).getSolver().getInverse();
    }

    /**
     * Print message for online simulation evaluation
     */
    public void print_sim_message(){
        double tmp_mean = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_max = 0;
        double tmp_mean_cov = 0;
        double tmp_min_cov = Double.MAX_VALUE;
        double tmp_max_cov = 0;
        double tmp_mean_prec2 = 0;
        double tmp_min_prec2 = Double.MAX_VALUE;
        double tmp_max_prec2 = 0;
        double tmp_mean_prec = 0;
        double tmp_min_prec = Double.MAX_VALUE;
        double tmp_max_prec = 0;
        for(int j = 0; j < this.P; j++){
            tmp_mean_prec2 += this.cov.getEntry(j, j);
            tmp_min_prec2 = Math.min(tmp_min_prec2, this.invcorr.getEntry(j, j));
            tmp_max_prec2 = Math.max(tmp_max_prec2,this.invcorr.getEntry(j, j));

            for(int jj = 0; jj < this.P; jj++){
                if (j == jj) {
                    continue;
                }

                tmp_mean += Math.abs(this.corr.getEntry(j, jj));
                tmp_mean_cov += Math.abs(this.cov.getEntry(j, jj));
                tmp_mean_prec += Math.abs(this.invcorr.getEntry(j, jj));

                tmp_min = Math.min(tmp_min, this.corr.getEntry(j, jj));
                tmp_max = Math.max(tmp_max, this.corr.getEntry(j, jj));
                tmp_min_cov = Math.min(tmp_min_cov, this.cov.getEntry(j, jj));
                tmp_max_cov = Math.max(tmp_max_cov, this.cov.getEntry(j, jj));
                tmp_min_prec = Math.min(tmp_min_prec, this.invcorr.getEntry(j, jj));
                tmp_max_prec = Math.max(tmp_max_prec, this.invcorr.getEntry(j, jj));
            }
        }
        tmp_mean /= (this.P * (this.P - 1) + 0.0);
        tmp_mean_cov /= (this.P * (this.P - 1) + 0.0);
        tmp_mean_prec /= (this.P * (this.P - 1) + 0.0);
        tmp_mean_prec2 /= (this.P  + 0.0);

        System.out.printf("Off-diagonal R    : abs mean: %.4f, min: %.4f, max: %.4f\n",
                tmp_mean, tmp_min, tmp_max);
        System.out.printf("Off-diagonal Sigma: abs mean: %.4f, min: %.4f, max: %.4f\n",
                tmp_mean_cov, tmp_min_cov, tmp_max_cov);
        System.out.printf("Diagonal InvR: mean: %.4f, min: %.4f, max: %.4f\n",
                tmp_mean_prec2, tmp_min_prec2, tmp_max_prec2);
        System.out.printf("Off-diagonal InvR: abs mean: %.4f, min: %.4f, max: %.4f\n",
                tmp_mean_prec, tmp_min_prec, tmp_max_prec);
        double det = new LUDecomposition(this.prec).getDeterminant();
        System.out.printf("Prec determinant: %.10f\n", Math.log(det));
        double det2 = new LUDecomposition(this.corr).getDeterminant();
        System.out.printf("R determinant: %.10f\n", Math.log(det2));
    }

}
