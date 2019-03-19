package sampler;

import cern.jet.random.tdouble.Exponential;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.correlation.Covariance;
import util.MathUtil;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by zehangli on 5/20/16.
 */
public class Mix_data {
    public String[] ID;
    public int N;
    public int N_test;
    public int P;
    public int G;
    public int[] type; // type of each dimension, 0 = continuous, 1 = binary, 2 = ordinal

    public double[] d;

    public int[] membership;
    public int[] membership_test;
    public int[] groupcount;
    public double[][] mu; // mu is the mean of Delta
    public double[][] Delta; // Delta is the mean of latent variable Z
    public double[][] Delta_expanded; // Delta_expanded is the mean of latent variable ZD

    public double[][] data;
    public double[][] latent;
    public double[][] expanded;

    // index set for removing one
    int[][] left_index;
    int[][] remove_index;


    boolean adaptive;
    public double a_sd0, b_sd0; // adaptive version
    public double[][] sd0;
    public double power; // modification to the original sssl prior

    // initialization
    public Mix_data(int N, int N_test, int P, int G){
        this.N = N;
        this.N_test = N_test;
        this.P = P;
        this.G = G;
        this.type = new int[P];
        this.d = new double[P];
        this.membership = new int[N];
        this.membership_test = new int[N];
        for(int i = 0; i < N; i++) membership_test[i] = G;
        this.groupcount = new int[G];
        this.mu = new double[G][P];
        this.Delta = new double[G][P];
        this.data = new double[N][P];
        this.latent = new double[N][P];
        this.init_group();

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
    }
    public Mix_data(int N, int P, int G){
        this.N = N;
        this.N_test = 0;
        this.P = P;
        this.G = G;
        this.type = new int[P];
        this.d = new double[P];
        this.membership = new int[N];
        this.membership_test = new int[N];
        for(int i = 0; i < N; i++) membership_test[i] = G;
        this.groupcount = new int[G];
        this.mu = new double[G][P];
        this.Delta = new double[G][P];
        this.data = new double[N][P];
        this.latent = new double[N][P];
        this.init_group();

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
    }

    // initiate adaptive sd0
    public void init_adaptive(double sd0, double a_sd0, double b_sd0, boolean adaptive, double power){
        this.sd0 = new double[this.G][this.P];
        for(int i = 0; i < this.G; i++){
            for(int j = 0; j < this.P; j++){
                this.sd0[i][j] = sd0;
            }
        }
        this.a_sd0 = a_sd0;
        this.b_sd0 = b_sd0;
        this.adaptive = adaptive;
        this.power = power;
    }

    // random group assignment
    public void init_group(){
        this.groupcount = new int[G];
        for(int i = 0; i < this.N; i++){
            this.membership[i] = i % G;
            this.groupcount[this.membership[i]] ++;
        }

    }

    // fixed group assignment
    public void init_group(int[] membership){
        this.groupcount = new int[G];
        for(int i = 0; i < this.N; i++){
            this.membership[i] = membership[i];
            this.groupcount[this.membership[i]] ++;
        }
    }

    public void update_groupcounts(boolean update_with_test){
        this.groupcount = new int[this.G];
        for(int i = 0; i < this.N; i++){
            if(update_with_test & this.membership[i] == G & this.membership_test[i] != G ){
                this.groupcount[this.membership_test[i]]++;
                continue;
            }
            if(this.membership[i] < G){
                this.groupcount[this.membership[i]]++;
                continue;
            }
            if(update_with_test & this.membership[i] == G & this.membership_test[i] == G){
                this.membership_test[i] = new Random().nextInt(G);
                this.groupcount[this.membership_test[i]]++;
                continue;
            }
        }
    }
    /**
     * add data from file Without Header but with ID
     * @param filename file name to read in data, comma separated, no header
     * @param marfilename file name to read in variable type information,
     *                    First line: 0 = cont, 1 = bin, 2 = ordinal
     *                    Second line: marginal mean of the latent variable
     * @param P dimension of data
     * @throws IOException
     */
    public void addData(String filename, String marfilename, int P) throws IOException {
        BufferedReader br0 = new BufferedReader(new FileReader(marfilename));
        // first line is type
        String[] line0 = br0.readLine().split(",");
        for(int j = 0; j < this.P; j++){
            this.type[j] = Integer.parseInt(line0[j]);
        }
        // todo: new method for data without 2nd line in marginal information
        // second line is marginal mean for latent variables
        line0 = br0.readLine().split(",");
        for(int j = 0; j < this.P; j++){
            this.mu[0][j] = Double.parseDouble(line0[j]);
        }

        BufferedReader br = new BufferedReader(new FileReader(filename));
        LinkedList<String[]> rows = new LinkedList<String[]>();
        this.P = P;
        this.N = 0;
        this.G = 1;

        String line;
        while ((line = br.readLine()) != null) {
            rows.addLast(line.split(","));
            this.N++;
        }
        br.close();

        this.ID = new String[this.N];
        this.data = new double[this.N][this.P];
        this.latent = new double[this.N][this.P];
        this.expanded = new double[this.N][this.P];
        this.Delta = new double[this.G][this.P];
        this.Delta_expanded = new double[this.G][this.P];

        for(int i = 0 ; i < this.N; i++){
            this.ID[i] = rows.get(i)[0];
            for(int j = 0; j < this.P; j++){
                this.data[i][j] = Double.parseDouble(rows.get(i)[j + 1]);
            }
        }
        System.out.println("Number of rows read from data: " + N + "\n");

    }




    // add latent data
    public void addLatent(double[][] latent){
        for(int i = 0; i < this.N; i ++){
            for(int j = 0; j < this.P; j++){
                this.latent[i][j] = latent[i][j];
            }
        }
    }

    /**
     * Sample marginal variance in PX model
     * @param rngG
     * @param invcorr
     */
    public void resample_D(Gamma rngG, RealMatrix invcorr){
        for(int i = 0; i < this.P; i++){
            // invcorr[i, i]/2 is the scale of invGamma,
            // so 2/invcorr[i, i] is the scale of Gamma,
            // so  invcorr[i, i]/2 is the rate of Gamma
            this.d[i] = 1 / rngG.nextDouble((this.P + 1.0)/2.0,
                    invcorr.getEntry(i, i)/2.0);
            this.d[i] = Math.sqrt(this.d[i]);
//            if(this.d[i] > 5){
//                System.out.println("Careful large expansion.");
//            }
        }
    }

    /**
     * Sample marginal variance in PX model
     * @param rngG
     * @param invcorr
     */
    public void resample_Dfix(Gamma rngG, RealMatrix invcorr){
        for(int i = 0; i < this.P; i++){
            // invcorr[i, i]/2 is the scale of invGamma,
            // so 2/invcorr[i, i] is the scale of Gamma,
            // so  invcorr[i, i]/2 is the rate of Gamma
            this.d[i] = 1 / rngG.nextDouble((this.P + 1.0)/2.0,
                    1/2.0);
//                                             invcorr.getEntry(i, i)/2.0);
            this.d[i] = Math.sqrt(this.d[i]);
            if(this.d[i] > 5){
                System.out.println("Careful large expansion.");
            }
        }
    }


    /**
     *  Sample marginal variance in SSSL model
     *
     * @param rngG
     * @param invcorr Inverse correlation matrix
     * @param inclusion inclusion indicator matrix
     * @param v0 slab
     * @param v1 spike
     * @param lambda prior
     */
    public void resample_D_seq(Gamma rngG, RealMatrix invcorr, int[][] inclusion, double v0, double v1, double
            lambda) {

        double v2;
        List<Integer> onetop = new ArrayList<>();
        for (int i = 0; i < this.P; i++) {
            onetop.add(i);
        }
        Collections.shuffle(onetop);
        for (int rep = 0; rep < 1; rep++) {
            double[] dnew = new double[this.P];
            for (int i : onetop) {
                double sum = 0;
                for (int j = 0; j < this.P; j++) {
                    if (j != i) {
                        v2 = inclusion[i][j] == 1 ? v1 * v1 : v0 * v0;
                        if(v2 != 0){
                            sum += invcorr.getEntry(i, j) * invcorr.getEntry(i, j) / (this.d[j] * this
                                    .d[j] * v2);
                        }
                    }
                }
                sum = lambda * invcorr.getEntry(i, i);
                this.d[i] = Math.sqrt( 1 / rngG.nextDouble(this.power + (this.P + 1.0) / 2.0,
                        sum / 2.0) );
            }
//            for (int i = 0; i < this.P; i++) this.d[i] = dnew[i];
        }
    }
    public void resample_D_carryover( RealMatrix cov){
        double maxvalue = Double.MAX_VALUE;
//        double sumsqare = 0;
//        for(int i = 0; i < this.P; i++) {
//            sumsqare += cov.getEntry(i,i) * cov.getEntry(i, i) / cov.getColumnDimension();
//        }
        double sumsqare = 1;
        for(int i = 0; i < this.P; i++) {
            if(cov.getEntry(i, i) > maxvalue){
                System.out.print(".");
            }
            if(this.type[i] == 0){
                // this.d[i] = 1;
                this.d[i] = Math.min(Math.sqrt(cov.getEntry(i, i) / sumsqare), maxvalue);
            }else if(this.type[i] == 1){
                this.d[i] = Math.min(Math.sqrt(cov.getEntry(i, i) / sumsqare), maxvalue);
            }else if(this.type[i] == 2){
                //todo: what to do for categorical case?
            }
        }
    }


    public void resample_D_normalize( RealMatrix cov){
        for(int i = 0; i < this.P; i++) {
            if(this.type[i] == 0){
                this.d[i] = 1.0 / Math.sqrt(cov.getEntry(i, i));
            }else if(this.type[i] == 1){
                this.d[i] = 1.0 / Math.sqrt(cov.getEntry(i, i));
            }else if(this.type[i] == 2){
                //todo: what to do for categorical case?
            }
        }
    }

    public void resample_D_lambda(Gamma rngG, RealMatrix invcorr, double lambda){
        for(int i = 0; i < this.P; i++){
            // invcorr[i, i]/2 is the scale of invGamma,
            // so 2/invcorr[i, i] is the scale of Gamma,
            // so  invcorr[i, i]/2 is the rate of Gamma
            this.d[i] = 1 / rngG.nextDouble((this.P + 1.0)/2.0,
                    lambda * invcorr.getEntry(i, i)/2.0);
            this.d[i] = Math.sqrt(this.d[i]);
            //this.d[i] = 1;
        }
    }


    /**
     *  Sample latent data
     * @param rand
     * @param rngN
     * @param rngE
     * @param Mean P-dimensional array of mean of the latent variables, truncation happens at 0.
     * @param Sigma P by P matrix of Covariance of the latent variables
     * @param verbose
     */
    public void resample_Z(Random rand, NormalDistribution rngN, Exponential rngE,
                           double[][] Mean, RealMatrix Sigma, boolean verbose) {

        // calculate list of inverse matrices Sigma[-j, -j] first
//        HashMap<Integer, RealMatrix> Sinv_times_Sigma_j_mj = new HashMap<>();
//        HashMap<Integer, RealMatrix> Sigma_j_mj = new HashMap<>();
//        for (int j = 0; j < this.P; j++) {
//            RealMatrix temp = MathUtil.colRemove(Sigma.getRowMatrix(j), j);
//            Sigma_j_mj.put(j, temp);
//            RealMatrix subSigma = MathUtil.subRemove(Sigma, j);
//            Sinv_times_Sigma_j_mj.put(j, new LUDecomposition(subSigma).getSolver().getInverse().preMultiply(temp));
//        }
        // calculate list of inverse matrices Sigma[-j, -j] first
        RealMatrix Sinv_times_Sigma_j_mj = new Array2DRowRealMatrix(new double[this.P][this.P-1]);
        HashMap<Integer, RealMatrix> Sigma_j_mj = new HashMap<>();
        HashMap<Integer, Double> Sigma_j_mj_Siv_Sigma_j_mj = new HashMap<>();
        for(int j = 0; j < this.P; j++){
            RealMatrix temp0 = MathUtil.colRemove(Sigma.getRowMatrix(j), j);
            Sigma_j_mj.put(j, temp0);
            RealMatrix subSigma = MathUtil.subRemove(Sigma, j);
            Sinv_times_Sigma_j_mj.setRow(j, (new LUDecomposition(subSigma).getSolver().getInverse().preMultiply(temp0)).getRow(0));
            Sigma_j_mj_Siv_Sigma_j_mj.put(j, ((temp0.transpose()).preMultiply(Sinv_times_Sigma_j_mj.getRowMatrix(j))).getEntry(0,0));
        }


        List<Integer> onetop = new ArrayList<>();
        for (int i = 0; i < this.P; i++) {
            onetop.add(i);
        }
        resample_Z_internal(rand, rngN, rngE, Mean, Sigma, verbose, onetop, Sinv_times_Sigma_j_mj, Sigma_j_mj);
    }

    public void resample_Z_internal(Random rand, NormalDistribution rngN, Exponential rngE,
                                    double[][] Mean, RealMatrix Sigma, boolean verbose, List<Integer> onetop,
                                    RealMatrix Sinv_times_Sigma_j_mj, HashMap<Integer, RealMatrix> Sigma_j_mj) {
        double tmp_max = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_mean = 0;
        double maxvalue = 5;//Double.MAX_VALUE;
        // Resample Z
        for(int i = 0; i < this.N; i++){
            //if(this.membership[i] == G) continue;
            int gtmp = this.membership[i];
            if(this.membership[i] == G ){
                gtmp = this.membership_test[i];
            }
            if(gtmp == G) continue;
            Collections.shuffle(onetop);
            for(int j : onetop){
                // continuous data case
                if(type[j] == 2){
                    // todo: categorical case - resample using ranks
                    continue;
                }
                double[][] Z_i_mj_minus_mu_mj = new double[this.P - 1][1];
                int counter = 0;
                for(int jj = 0; jj < this.P; jj ++ ){
                    if(jj == j){continue;}
                    Z_i_mj_minus_mu_mj[counter][0] = this.latent[i][jj] - Mean[gtmp][jj]; counter ++;
                }
                RealMatrix diff_mean = new Array2DRowRealMatrix(Z_i_mj_minus_mu_mj);

                RealMatrix pre = Sinv_times_Sigma_j_mj.getRowMatrix(j);  //Sinv.get(j).preMultiply(Sigma_j_mj.get(j));
                RealMatrix change_temp = diff_mean.preMultiply(pre);
                RealMatrix change_temp2 = (Sigma_j_mj.get(j).transpose()).preMultiply(pre);

                double mean_temp = Mean[gtmp][j] + change_temp.getEntry(0, 0);
                double sd2_temp = Sigma.getEntry(j, j) - change_temp2.getEntry(0, 0);
                double sd_temp = Math.sqrt(sd2_temp);

                // todo: truncated domain needs re-derived for cont and categorical data
                if(type[j] == 1){
                    double min = (this.data[i][j] == 1 & type[j] == 1) ? 0 : maxvalue * (-1);
                    double max = (this.data[i][j] == 0 & type[j] == 1) ? 0 : maxvalue;
                    this.latent[i][j] = MathUtil.truncNormal(rand, rngN, rngE,
                            mean_temp, sd_temp, min, max, maxvalue);
                    if(this.latent[i][j] < -100000){
                        System.out.println(".");
                    }
                    if(Math.abs(this.latent[i][j]) > 6){
                        //System.out.println(".");
                    }
                }else if(type[j] == 0 & this.data[i][j] == -Double.MAX_VALUE){
                    this.latent[i][j] = MathUtil.truncNormal(rand, rngN, rngE,
                            mean_temp, sd_temp, -maxvalue, maxvalue, maxvalue);
                }

            }
        }

        if(verbose){
            for(int i = 0; i < this.N; i++){
                for(int j = 0; j < this.P; j++) {
                    tmp_mean += this.latent[i][j] / (this.N * this.P + 0.0);
                    tmp_min = Math.min(tmp_min, this.latent[i][j]);
                    tmp_max = Math.max(tmp_max, this.latent[i][j]);
                }
            }
            System.out.printf("Finish latent variables re-sample: mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean, tmp_min, tmp_max);
        }


    }
    public double[][] getdomain(Latent_model model, int i, Set<Integer> active, double[][] Mean, double maxvalue){
        double[][] joint = new double[this.P][2];
        for(int j = 0; j < this.P; j++){
            if(model.data.data[i][j] != -Double.MAX_VALUE & model.data.type[j] == 1){
                joint[j][0] = maxvalue;
                joint[j][1] = maxvalue * (-1);
                for(int g : active){
                    if(this.data[i][j] == 0){
                        joint[j][0] = maxvalue * (-1);
                        joint[j][1] = Math.max(joint[j][1],  -Mean[g][j]);
                    }else{
                        joint[j][0] = Math.min(joint[j][0],  -Mean[g][j]);
                        joint[j][1] = maxvalue;
                    }
                }
            }else {
                joint[j][0] = maxvalue * (-1);
                joint[j][1] = maxvalue;
            }
        }
        return(joint);
    }
    public double[][] resample_Z_monte(Random rand, NormalDistribution rngN, Exponential rngE,
                                       double[][] Mean, RealMatrix Sigma, boolean verbose, int R,
                                       Latent_classifier model, boolean update_prob, Gamma rngG,
                                       boolean same_pop, double temp){

        double[][] out = new double[N_test][G];
        LUDecomposition solver = new LUDecomposition(Sigma);
        RealMatrix invcorr = solver.getSolver().getInverse();
        double det = solver.getDeterminant();
        double maxvalue = 5;



        List<Integer> onetop = new ArrayList<>();
        for (int i = 0; i < this.P; i++) {
            onetop.add(i);
        }

        // Resample new membership probabilities
//        long startTime = System.nanoTime();
//        long part1=0;
//        long part2=0;
//        long part3=0;
        if (R > 0) {
            int counter = 0;
            for(int i = 0; i < this.N_test; i++) {
                double[] zerovec = new double[this.G];
                double[] prob = new double[this.G];
                for (int g = 0; g < this.G; g++) {
                    if (!model.impossible[i][g]) {
                        zerovec[g] = 1;
                    }
                }
                if(model.test_case[i]){
                    // get continuous density
                    int nonmissing_binary = 0;
                    int nonmissing_cont = 0;
                    for(int j=0; j < this.P; j++){
                        if(model.data.data[i][j] != - Double.MAX_VALUE & model.data.type[j] == 1)  nonmissing_binary++;
                        if(model.data.data[i][j] != - Double.MAX_VALUE & model.data.type[j] == 0)  nonmissing_cont++;
                    }
                    double maxlog = -Double.MAX_VALUE;
                    if(nonmissing_cont > 0) {
                        double[] x = new double[nonmissing_cont];
                        double[][] meansub = new double[this.G][nonmissing_cont];
                        int[] remain = new int[nonmissing_cont];
                        int counter1 = 0;
                        for (int j = 0; j < this.P; j++) {
                            if (model.data.data[i][j] != -Double.MAX_VALUE & model.data.type[j] == 0) {
                                x[counter1] = model.data.latent[i][j];
                                for (int g = 0; g < this.G; g++) meansub[g][counter1] = Mean[g][j];
                                remain[counter1] = j;
                                counter1++;
                            }
                        }
                        RealMatrix invcorrsub = new LUDecomposition(Sigma.getSubMatrix(remain, remain)).getSolver().getInverse();
                        for (int g = 0; g < this.G; g++) {
                            if (model.impossible[i][g]) {
                                continue;
                            }
                            prob[g] = Math.log(model.post_prob[g]);
                            double add = MathUtil.multivariate_normal_logdensity(x, meansub[g], invcorrsub, 1);
                            prob[g] += add;
                            maxlog = Math.max(prob[g], maxlog);
                        }
                    }
                    if(nonmissing_binary > 0){
                        double[][] drawmatrix = new double[this.G][R]; // initial with all 0.0, final active changed to 1.0

//                        double[] ztemp = new double[this.P];
//                        for(int j = 0; j < this.P; j++){
//                            ztemp[j] = this.latent[i][j] - Mean[this.membership_test[i]][j];
//                        }
                        // initial joint domain
//                        Set<Integer> active = new HashSet<>();
//                        for(int g = 0; g < this.G; g++){
//                             boolean remove = false;
//                              for(int j = 0; j < this.P; j++) {
//                                  remove = (this.data[i][j] == 0 & ztemp[j] > -Mean[g][j]) | (this.data[i][j] == 1 & ztemp[j] < -Mean[g][j]);
//                                  if(remove){
//                                      active.remove(g);
//                                  }
//                              }
//                              if(!remove){
//                                active.add(g);
//                              }
//                        }
//                        HashMap<Integer, Set<Integer>> outside_coord = new HashMap<>();
//                        for(int g = 0; g < this.G; g++){
//                            outside_coord.put(g, new HashSet<>());
//                        }
//                        double[][] joint = getdomain(model, i, active, Mean,  maxvalue);

                        int counter1 = 0;

                        for (int rr = 0; rr < R; rr++) {
                            double[] ztemp = new double[this.P];
                            Set<Integer> active = new HashSet<>();
                            for(int g = 0; g < this.G; g++){
                                active.add(g);
                            }
                            HashMap<Integer, Set<Integer>> outside_coord = new HashMap<>();
                            for(int g = 0; g < this.G; g++){
                                outside_coord.put(g, new HashSet<>());
                            }
                            double[][] joint = getdomain(model, i, active, Mean,  maxvalue);

                            Collections.shuffle(onetop);
                            int j = onetop.get(0);
                            ztemp[j] = MathUtil.truncNormal(rand, rngN, rngE,
                                    0, Math.sqrt(Sigma.getEntry(j, j)), joint[j][0], joint[j][1], maxvalue);
                            for (int jj = 1; jj < this.P; jj++) {
                                j = onetop.get(jj);
                                if(model.data.type[j] == 1) {
                                    double mean_temp = 0;

                                    RealMatrix Sinv_times_Sigma_j_mj;
                                    double[] Sigma_j_mj = new double[jj];
                                    double Sigma_j_mj_Siv_Sigma_j_mj = 0;
                                    int counter0 = 0;
                                    for(int jtmp : onetop.subList(0, jj-1)) {
                                        Sigma_j_mj[counter0] = Sigma.getEntry(j, jtmp);
                                        counter0++;
                                    }
                                    RealMatrix tmp0 = new Array2DRowRealMatrix(Sigma_j_mj);
                                    int[] subarray = new int[jj];
                                    for(int jtmp = 0; jtmp < jj; jtmp++) subarray[jtmp] = onetop.get(jtmp);
                                    RealMatrix subSigma = Sigma.getSubMatrix(subarray, subarray);
                                    RealMatrix tmp = new LUDecomposition(subSigma).getSolver().getInverse().preMultiply(tmp0.transpose());
                                    Sinv_times_Sigma_j_mj = tmp.getRowMatrix(0);
                                    Sigma_j_mj_Siv_Sigma_j_mj = ((tmp0.transpose()).preMultiply(Sinv_times_Sigma_j_mj.transpose())).getEntry(0,0);

//                                    RealMatrix Sinv_times_Sigma_j_mj = new Array2DRowRealMatrix(new double[this.P][this.P-1]);
//                                    HashMap<Integer, RealMatrix> Sigma_j_mj = new HashMap<>();
//                                    HashMap<Integer, Double> Sigma_j_mj_Siv_Sigma_j_mj = new HashMap<>();
//                                    for(int j = 0; j < this.P; j++){
//                                        RealMatrix temp0 = MathUtil.colRemove(Sigma.getRowMatrix(j), j);
//                                        Sigma_j_mj.put(j, temp0);
//                                        RealMatrix subSigma = MathUtil.subRemove(Sigma, j);
//                                        Sinv_times_Sigma_j_mj.setRow(j, (new LUDecomposition(subSigma).getSolver().getInverse().preMultiply(temp0)).getRow(0));
//                                        Sigma_j_mj_Siv_Sigma_j_mj.put(j, ((temp0.transpose()).preMultiply(Sinv_times_Sigma_j_mj.getRowMatrix(j))).getEntry(0,0));
//                                    }

                                    counter0 = 0;
                                    for(int jtmp : onetop.subList(0, jj-1)){
                                        mean_temp += ztemp[jtmp] * Sinv_times_Sigma_j_mj.getEntry(0, counter0);
                                        counter0 ++;
                                    }
                                    double sd_temp = Math.sqrt(Sigma.getEntry(j, j) - Sigma_j_mj_Siv_Sigma_j_mj);
                                    ztemp[j] = MathUtil.truncNormal(rand, rngN, rngE,
                                            mean_temp, sd_temp, joint[j][0], joint[j][1], maxvalue);
                                    // remove excluded causes
                                    boolean changed = false;
                                    for(int g = 0; g < this.G; g++){
                                        boolean remove = (this.data[i][j] == 0 & ztemp[j] > -Mean[g][j]) | (this.data[i][j] == 1 & ztemp[j] < -Mean[g][j]);
                                        if(remove){
                                            outside_coord.get(g).add(j);
                                            active.remove(g);
                                            changed = true;
                                         // if satisfy the condition
                                        }else if(outside_coord.get(g).contains(j)){
                                            outside_coord.get(g).remove(j);
                                            if(outside_coord.get(g).size() == 0){
                                                active.add(g);
                                                changed = true;
                                            }

                                        }
                                    }
                                    if(active.size() == 0){
                                        System.out.printf("no active set!");
                                    }
                                    if(active.size() == 1){
                                        break;
                                    }
                                    if(changed) joint = getdomain(model, i, active, Mean,  maxvalue);
                                }
                            }
                            for(int g : active){
                                drawmatrix[g][rr] = 1;
                            }
                        }

                        for(int g = 0; g < this.G; g++){
                            double tmp = 0;
                            for(int r = 0; r < drawmatrix[0].length; r++) tmp += (drawmatrix[g][r]) / (R + 0.0);
                            prob[g] += Math.log(tmp);
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
        }
        System.out.println("Finishing resampling membership");
        if(update_prob) model.update_group_prob(out, rand, rngG, same_pop, temp);
        System.out.println("Finishing resampling Fraction Distribution");

//        resample_Z_internal(rand, rngN, rngE, Mean, Sigma, verbose, onetop, Sinv_times_Sigma_j_mj, Sigma_j_mj);
        resample_Z(rand, rngN, rngE, Mean, Sigma, verbose);
        return(out);
    }


    // initialize Delta
    public void initial_Delta(double[][] mean){
        for(int i = 0; i < this.G; i++){
            for(int j = 0; j < this.P; j++){
                this.Delta[i][j] = mean[i][j];
                this.d[j] = 1;
            }
        }
    }

    // initialize Z
    public void initial_Z(Random rand, NormalDistribution rngN, Exponential rngE,
                          double[][] mu, double[] sd, boolean verbose){
        double tmp_max = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_mean = 0;
        double maxvalue = 5;

        for(int i = 0; i < this.N; i++){
            int member = this.membership[i];
            if(member == this.G){
                member = this.membership_test[i];
            }
//            if(member == this.G){
//                member = rand.nextInt(G);
//                this.membership_test[i] = member;
//            }
            for(int j = 0; j < this.P; j++){
                // continuous data case
                if(type[j] == 0){
                    // todo: does it make sense to set missing to 0?
                    this.latent[i][j] = this.data[i][j] == -Double.MAX_VALUE ? 0 : this.data[i][j];

                    continue;
                }else if(type[j] == 1){
                    double min = (this.data[i][j] == 1 & type[j] == 1) ? 0 : maxvalue * (-1);
                    double max = (this.data[i][j] == 0 & type[j] == 1) ? 0 : maxvalue;
                    if(member == this.G){
                        this.latent[i][j] = this.data[i][j] == 1 ? 1 : 0;
                        this.latent[i][j] = this.data[i][j] == 0 ? -1 : this.data[i][j];
                    }else{
                        this.latent[i][j] = MathUtil.truncNormal(rand, rngN, rngE, mu[member][j],
                                sd[j], min, max, maxvalue);
                    }

                    //this.latent[i][j] =  mu[this.membership[i]][j];

                }else if(type[j] == 2){
                    // todo: categorical case: initialize latent variable

                }
            }
        }
        if(verbose){
            for(int i = 0; i < this.N; i++){
                for(int j = 0; j < this.P; j++){
                    tmp_mean += this.latent[i][j] / (this.N * this.P + 0.0);
                    tmp_min = Math.min(tmp_min, this.latent[i][j]);
                    tmp_max = Math.max(tmp_max, this.latent[i][j]);
                }
            }
            System.out.printf("Initial latent variables : mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean, tmp_min, tmp_max);
        }
    }

    /** compute statistics for expanded parameter
     *
     * @param gID group ID
     * @return apply(expanded, 2, mean)
     */
    public RealMatrix computeWbar(int gID, boolean update_with_test){
        double[][] wbar = new double[this.P][1];
        for(int i = 0; i < this.N; i++) {
            if(this.membership[i] == gID | (update_with_test & this.membership_test[i] == gID)){
                for (int j = 0; j < this.P; j++) {
                    wbar[j][0] += this.expanded[i][j] / (this.groupcount[gID] + 0.0);
                }
            }
        }
        return(new Array2DRowRealMatrix(wbar));
    }

    public RealMatrix computeWprodbyGroup(boolean update_with_test, double[] tau){
        return(computeWprodbyGroup(update_with_test, tau, false));
    }
    public RealMatrix computeWprodbyGroup(boolean update_with_test, double[] tau, boolean verbose){
        RealMatrix Wprodsum = new Array2DRowRealMatrix(new double[this.P][this.P]);

        for(int i = 0; i < this.G; i++){
            // do this for each group
            if(this.groupcount[i] <= 0){continue;}

            double[][] expanded_sub = new double[groupcount[i]][P];
            int counter = 0;
            for(int ii = 0; ii < this.N; ii++){
                if(this.membership[ii] == i | (update_with_test & this.membership_test[ii] == i)){
                    for(int j = 0; j < this.P; j++){
                        expanded_sub[counter][j] = (this.expanded[ii][j] - this.Delta_expanded[i][j]) / tau[j];
                    }
                    counter ++;
                }
            }
            RealMatrix expanded_sub_mat = new Array2DRowRealMatrix(expanded_sub);
            RealMatrix Wprod = expanded_sub_mat.preMultiply(expanded_sub_mat.transpose());
            Wprodsum = Wprodsum.add(Wprod);
        }
        if(verbose){
            double tmp_mean = 0;
            double tmp_max = 0;
            for(int i = 0; i < this.P; i++){
                tmp_mean += Wprodsum.getEntry(i, i)/(this.P+0.0);
                tmp_max = Math.max(tmp_max,  Wprodsum.getEntry(i, i));
            }
            System.out.printf("S matrix : mean: %.4f, max: %.4f\n",
                    tmp_mean, tmp_max);
        }
        return(Wprodsum);
    }

    public RealMatrix computeWprodbyGroup_conj(boolean update_with_test, double[] tau){
        // get the first part
        RealMatrix Wprodsum = computeWprodbyGroup(update_with_test, tau);

        // add the part from prior
        for(int i = 0; i < this.G; i++){

            if(this.groupcount[i] <= 0){continue;}

            double[][] expanded_sub = new double[1][P];
            for(int j = 0; j < this.P; j++){
                expanded_sub[0][j] = (this.Delta_expanded[i][j] - this.mu[i][j] * this.d[j]) / tau[j];
            }
            RealMatrix expanded_sub_mat = new Array2DRowRealMatrix(expanded_sub);
            RealMatrix Wprod = expanded_sub_mat.preMultiply(expanded_sub_mat.transpose()).scalarMultiply(1/this
                    .sd0[i][0] / this.sd0[i][0]);
            Wprodsum = Wprodsum.add(Wprod);
        }
        return(Wprodsum);
    }

    public RealMatrix computeSprodbyGroup(boolean update_with_test){
        RealMatrix Sprodsum = new Array2DRowRealMatrix(new double[this.P][this.P]);
        double size = -1;
        for(int i = 0; i < this.G; i++) {
            if (this.groupcount[i] <= 0) {
                continue;
            }

            double[][] latent_sub = new double[groupcount[i]][P];
            int counter = 0;
            for (int ii = 0; ii < this.N; ii++) {
                if (this.membership[ii] == i | (update_with_test & this.membership_test[ii] == i)) {
                    for (int j = 0; j < this.P; j++) {
                        latent_sub[counter][j] = this.latent[ii][j] - this.Delta[i][j];
                    }
                    counter++;
                    size++;
                }
            }
            RealMatrix expanded_sub_mat = new Array2DRowRealMatrix(latent_sub);
            RealMatrix Wprod = expanded_sub_mat.preMultiply(expanded_sub_mat.transpose());
            Sprodsum = Sprodsum.add(Wprod);
        }
        Sprodsum = Sprodsum.scalarMultiply(1/size);
        return(Sprodsum);
    }

    /** compute statistics for expanded parameter
     *
     * @param verbose
     * @return Cov(X) + I_p, notice cov calculat
     * ed with N instead of N-1
     */
    public RealMatrix computeWprodPlusIp(boolean verbose, boolean update_with_test, double[] tau){
        double tmp_max = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_mean = 0;

        RealMatrix Wprod = this.computeWprodbyGroup(update_with_test, tau);

        for (int j = 0; j < this.P; j++) {
            Wprod.addToEntry(j, j, 1.0);
        }

        if(verbose){
            tmp_max = 0;
            tmp_min = Double.MAX_VALUE;
            tmp_mean = 0;
            for(int j = 0; j < this.P; j++){
                for(int k = 0; k < this.P; k++){
                    tmp_mean += Wprod.getEntry(j, k) / (this.P * this.P + 0.0);
                    tmp_min = Math.min(tmp_min, Wprod.getEntry(j, k));
                    tmp_max = Math.max(tmp_max, Wprod.getEntry(j, k));
                }
            }
            System.out.printf("Finish Calculating Wprod+Ip : mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean, tmp_min, tmp_max);
        }
        return(Wprod);
    }

    public RealMatrix computeWprodPlusIp_conj(boolean verbose, boolean update_with_test, double[] tau){
        double tmp_max = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_mean = 0;

        RealMatrix Wprod = this.computeWprodbyGroup_conj(update_with_test, tau);

        for (int j = 0; j < this.P; j++) {
            Wprod.addToEntry(j, j, 1.0);
        }

        if(verbose){
            tmp_max = 0;
            tmp_min = Double.MAX_VALUE;
            tmp_mean = 0;
            for(int j = 0; j < this.P; j++){
                for(int k = 0; k < this.P; k++){
                    tmp_mean += Wprod.getEntry(j, k) / (this.P * this.P + 0.0);
                    tmp_min = Math.min(tmp_min, Wprod.getEntry(j, k));
                    tmp_max = Math.max(tmp_max, Wprod.getEntry(j, k));
                }
            }
            System.out.printf("Finish Calculating Wprod+Ip : mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean, tmp_min, tmp_max);
        }
        return(Wprod);
    }

    /**
     * Expand Delta and latent by a vector
     *
     * @param d vector of length P
     * @param verbose
     */
    public void expand(double[] d, boolean verbose){
        double tmp_mean = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_max = 0;
        double tmp_mean_e = 0;
        double tmp_min_e = Double.MAX_VALUE;
        double tmp_max_e = 0;

        for(int i = 0; i < this.G; i++){
            for(int j = 0; j < this.P; j++){
                this.Delta_expanded[i][j] = this.Delta[i][j] * d[j];
                tmp_mean += d[j]/(this.P * this.G +0.0);
                tmp_min = Math.min(tmp_min, d[j]);
                tmp_max = Math.max(tmp_max, d[j]);
            }
        }
        for(int i = 0; i < this.N; i ++){
            for(int j = 0; j < this.P; j++){
                this.expanded[i][j] = this.latent[i][j] * d[j];
                tmp_mean_e += this.expanded[i][j]/(this.P * this.N +0.0);
                tmp_min_e = Math.min(tmp_min_e, this.expanded[i][j]);
                tmp_max_e = Math.max(tmp_max_e, this.expanded[i][j]);
            }
        }
        if(verbose){
            System.out.printf("Expansion parameter (sd): mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean, tmp_min, tmp_max);
            System.out.printf("Expanded latent variables: mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean_e, tmp_min_e, tmp_max_e);
        }
    }

    /**
     * Update Delta from new expanded vector
     *
     * @param cov Update Delta by expanding it with diagonal elements of a matrix
     * @param verbose
     */
    public void update_Delta(RealMatrix cov, boolean verbose){
        double tmp_mean = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_max = 0;
        double tmp_mae = 0;
        double tmp_mse = 0;
        double tmp_mean_d = 0;
        double tmp_min_d = Double.MAX_VALUE;
        double tmp_max_d = 0;
        /** Update P(S|C) or Delta for each group separately **/

        for(int j = 0; j < this.P; j++){
            this.d[j] = Math.sqrt(cov.getEntry(j, j));
            for(int i = 0; i < this.G; i++){
//                if(this.groupcount[i] <= 0) continue;
                this.Delta[i][j] = this.Delta_expanded[i][j] / this.d[j];

                tmp_mean += this.Delta[i][j] / (this.P + 0.0);
                tmp_min = Math.min(tmp_min, this.Delta[i][j]);
                tmp_max = Math.max(tmp_max, this.Delta[i][j]);

                tmp_mae += Math.abs(this.Delta[i][j] - this.mu[i][j]) / (this.P*this.G + 0.0);
                tmp_mse += Math.pow(this.Delta[i][j] - this.mu[i][j], 2) / (this.P*this.G + 0.0);
            }
        }
        for(int i = 0; i < this.P; i++){
            tmp_mean_d += this.d[i] / (this.P + 0.0);
            tmp_min_d = Math.min(tmp_min_d, this.d[i]);
            tmp_max_d = Math.max(tmp_max_d, this.d[i]);
        }

        if(verbose){
            System.out.printf("+Induced expansion parameter: mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean_d, tmp_min_d, tmp_max_d);
            System.out.printf("Mean (delta): mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean, tmp_min, tmp_max);
            System.out.printf("Delta deviation from prior: MAE: %.4f, MSE: %.4f\n",
                    tmp_mae, tmp_mse);
        }
    }

    public void update_Delta(double[] d, boolean verbose){
        double tmp_mean = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_max = 0;
        double tmp_mae = 0;
        double tmp_mse = 0;
        double tmp_mean_d = 0;
        double tmp_min_d = Double.MAX_VALUE;
        double tmp_max_d = 0;
        /** Update P(S|C) or Delta for each group separately **/

        for(int j = 0; j < this.P; j++){
            for(int i = 0; i < this.G; i++){
//                if(this.groupcount[i] <= 0) continue;
                this.Delta[i][j] = this.Delta_expanded[i][j] / d[j];

                tmp_mean += this.Delta[i][j] / (this.P + 0.0);
                tmp_min = Math.min(tmp_min, this.Delta[i][j]);
                tmp_max = Math.max(tmp_max, this.Delta[i][j]);

                tmp_mae += Math.abs(this.Delta[i][j] - this.mu[i][j]) / (this.P*this.G + 0.0);
                tmp_mse += Math.pow(this.Delta[i][j] - this.mu[i][j], 2) / (this.P*this.G + 0.0);
            }
        }
        if(verbose){
            System.out.printf("Mean (delta): mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean, tmp_min, tmp_max);
            System.out.printf("Delta deviation from prior: MAE: %.4f, MSE: %.4f\n",
                    tmp_mae, tmp_mse);
        }
    }

    /**
     * Resample expanded mean vector
     *
     * @param sd0 standard deviation of the latent Gaussian variables
     * @param prec The Precision matrix
     * @param rngEngine
     * @param verbose
     */
    // tildeR^-1 = Lambda^-1 D Omega D Lambda^-1
    public void resample_expanded(double[][] sd0, RealMatrix prec, double[] tau,  RandomGenerator rngEngine, boolean
            verbose, boolean
                                          update_with_test) {

        double tmp_mean = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_max = 0;

        for(int i = 0; i < this.G; i++) {

            if(this.groupcount[i] <= 0){
                continue;
            }
            double[][] tildeDinv = new double[this.P][this.P];
            double[][] Lambdainv = new double[this.P][this.P];
            for (int j = 0; j < this.P; j++) {
                tildeDinv[j][j] = 1 / (this.d[j] * this.d[j] * sd0[i][j] * sd0[i][j]);
                Lambdainv[j][j] = 1/tau[j];
            }

            RealMatrix preterm = new Array2DRowRealMatrix(tildeDinv);
            RealMatrix Lambdainvmat = new Array2DRowRealMatrix(Lambdainv);
            preterm = preterm.scalarMultiply(1/(this.groupcount[i] + 0.0));
            preterm = preterm.add(Lambdainvmat.preMultiply(prec).preMultiply(Lambdainvmat));

            RealMatrix tildeSigma = new LUDecomposition(preterm).getSolver().getInverse();
            tildeSigma = tildeSigma.scalarMultiply(1/(this.groupcount[i] + 0.0));
            tildeSigma = (tildeSigma.add(tildeSigma.transpose())).scalarMultiply(0.5);

            double[][] DinvMu = new double[this.P][1];
            for (int j = 0; j < this.P; j++) {
                DinvMu[j][0] = this.mu[i][j] / (this.d[j] * sd0[i][j] * sd0[i][j]);
            }

            RealMatrix post0 = new Array2DRowRealMatrix(DinvMu);
            RealMatrix Wbar = this.computeWbar(i, update_with_test);
            RealMatrix SigmaInv_w_bar = Wbar.preMultiply(Lambdainvmat.preMultiply(prec).preMultiply(Lambdainvmat));
            RealMatrix post1 = SigmaInv_w_bar.scalarMultiply(this.groupcount[i] + 0.0);
            RealMatrix post = post0.add(post1);

            RealMatrix mu_matrix = post.preMultiply(tildeSigma);
            double[] mu1 = (mu_matrix.transpose()).getData()[0];

            NormalDistribution rngN = new NormalDistribution(0, 1);
            // double[][] sigma1 = tildeSigma.getData();
            // MultivariateNormalDistribution rngMNV = new MultivariateNormalDistribution(rngEngine, mu1, sigma1);
            // double[] DeltaNew = rngMNV.sample();
            // using Chol decomposition directly,
            // Sigma = LL^T, z ~ N(0, I) then (mu + Lz) gives N(mu, Sigma)
            RealMatrix L = new CholeskyDecomposition(tildeSigma).getL();
            RealMatrix z = new Array2DRowRealMatrix(rngN.sample(P));
            double[]  DeltaNew = ((z.preMultiply(L)).add(mu_matrix)).transpose().getData()[0];


            for (int j = 0; j < this.P; j++) {
                this.Delta_expanded[i][j] = DeltaNew[j];
//                if(Math.abs(DeltaNew[j]) / this.d[j] > 6){
//                    System.out.println("Delta too big");
//                }
                tmp_mean += DeltaNew[j] / (this.P*this.G + 0.0);
                tmp_min = Math.min(tmp_min, DeltaNew[j]);
                tmp_max = Math.max(tmp_max, DeltaNew[j]);
            }
        }
        if (verbose) {
            System.out.printf("Expanded Mean (gamma): mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean, tmp_min, tmp_max);
        }

    }

    public void resample_expanded_conj(double[][] sd0, RealMatrix cov, RandomGenerator rngEngine, boolean verbose,
                                       boolean
                                               update_with_test) {

        double tmp_mean = 0;
        double tmp_min = Double.MAX_VALUE;
        double tmp_max = 0;

        for(int i = 0; i < this.G; i++) {

            if(this.groupcount[i] <= 0){
                continue;
            }

            RealMatrix mu_matrix = new Array2DRowRealMatrix(this.mu[i]);
            RealMatrix Wbar = this.computeWbar(i, update_with_test);
            double ratio = (1 / sd0[i][0]/sd0[i][0]) / (1 / sd0[i][0]/sd0[i][0] + groupcount[i]);
            for(int j = 0; j < this.P; j++){
                mu_matrix.setEntry(j, 0, this.mu[i][j] * this.d[j] * ratio + Wbar.getEntry(j, 0) * (1 - ratio));
            }

            RealMatrix tildeSigma =  cov.scalarMultiply( 1.0 / (this.groupcount[i] + 1/sd0[i][0]/sd0[i][0]));
            tildeSigma = tildeSigma.add(tildeSigma.transpose()).scalarMultiply(0.5);
            NormalDistribution rngN = new NormalDistribution(0, 1);
            // double[][] sigma1 = tildeSigma.getData();
            // MultivariateNormalDistribution rngMNV = new MultivariateNormalDistribution(rngEngine, mu1, sigma1);
            // double[] DeltaNew = rngMNV.sample();
            // using Chol decomposition directly,
            // Sigma = LL^T, z ~ N(0, I) then (mu + Lz) gives N(mu, Sigma)
            RealMatrix L = new CholeskyDecomposition(tildeSigma).getL();
            RealMatrix z = new Array2DRowRealMatrix(rngN.sample(P));
            double[]  DeltaNew = ((z.preMultiply(L)).add(mu_matrix)).transpose().getData()[0];


            for (int j = 0; j < this.P; j++) {
                this.Delta_expanded[i][j] = DeltaNew[j];
                tmp_mean += DeltaNew[j] / (this.P*this.G + 0.0);
                tmp_min = Math.min(tmp_min, DeltaNew[j]);
                tmp_max = Math.max(tmp_max, DeltaNew[j]);
            }
        }
        if (verbose) {
            System.out.printf("Expanded Mean (gamma): mean: %.4f, min: %.4f, max: %.4f\n",
                    tmp_mean, tmp_min, tmp_max);
        }
    }

    public void resample_adaptivesd0(Gamma rngG, boolean verbose){
        // adaptively change also the sd0
        double a = this.a_sd0 + 0.5 * this.P;
        double[] b = new double[this.G];
        double[] sdout = new double[this.G];
        for(int i = 0; i < this.G; i++){
            b[i] = this.b_sd0;
            for(int j = 0; j < this.P; j++){
                b[i] +=  0.5 * Math.pow(this.Delta[i][j] - this.mu[i][j], 2);
            }
            double sd0_square = 1 / rngG.nextDouble(a, b[i]);
            for(int j = 0; j < this.P; j++) this.sd0[i][j] = Math.sqrt(sd0_square);
            sdout[i] = this.sd0[i][0];
        }
        if(verbose){
            System.out.printf("Resampled SD0 "+ Arrays.toString(sdout) + "\n");
        }

    }



    public void resample_adaptivesd0_by_symp(Gamma rngG, boolean verbose){
        // adaptively change also the sd0
        double a = this.a_sd0 + 0.5 * this.G;
        double[] b = new double[this.P];
        double meansd = 0;
        for(int j = 0; j < this.P; j++){
            b[j] = this.b_sd0;
            for(int i = 0; i < this.G; i++){
                // otherwise the Delta_expanded does not make sense
                if(this.groupcount[i] > 0)  b[j] +=  0.5 * Math.pow(this.Delta[i][j] - this
                        .mu[i][j], 2);
            }
            double sd0_square = 1 / rngG.nextDouble(a, b[j]);
            for(int i = 0; i < this.G; i++) {
                this.sd0[i][j] = Math.sqrt(sd0_square);
            }
            meansd += sd0_square / (this.P + 0.0);
        }
        if(verbose){
            System.out.printf("Resampled SD0 "+ meansd + "\n");
        }

    }

    public void resample_adaptivesd0_conj_single(Gamma rngG, RealMatrix invcor, boolean verbose){
        // adaptively change also the sd0
        double a = this.a_sd0 + 0.5 * this.G;
        double b = this.b_sd0;
        double sd = 0;
        RealMatrix delta_minus_mu;
        for(int i = 0; i < this.G; i++) {
            // compute (delta_c - mu_c)^T R^{-1} (delta_c - mu_c)
            delta_minus_mu = new Array2DRowRealMatrix(this.Delta_expanded[i]);
            for (int j = 0; j < this.P; j++)
                delta_minus_mu.setEntry(j, 0, this.Delta_expanded[i][j] / this.d[j] - this
                        .mu[i][j]);
            b += 0.5 * delta_minus_mu.preMultiply(invcor).preMultiply(delta_minus_mu.transpose()).getEntry(0, 0);
        }
        sd = Math.sqrt(1 / rngG.nextDouble(a, b));
        for(int i = 0; i < this.G; i++) {
            for(int j = 0; j < this.P; j++){
                this.sd0[i][j] = sd;
            }
        }
        if(verbose){
            System.out.printf("Resampled SD0       %.4f\n", sd);
        }

    }


    public void resample_adaptivesd0_conj(Gamma rngG, RealMatrix invcor, boolean verbose){
        // adaptively change also the sd0
        double a = this.a_sd0 + 0.5;
        double[] b = new double[this.G];
        double[] meansd = new double[this.G];
        RealMatrix delta_minus_mu;
        for(int i = 0; i < this.G; i++){
            b[i] = this.b_sd0;
            // compute (delta_c - mu_c)^T R^{-1} (delta_c - mu_c)
            delta_minus_mu = new Array2DRowRealMatrix( this.Delta_expanded[i]);
            for(int j=0; j < this.P; j++) delta_minus_mu.setEntry(j, 0 , this.Delta_expanded[i][j]/ this.d[i] - this
                    .mu[i][j]);
            b[i] += 0.5 * delta_minus_mu.preMultiply(invcor).preMultiply(delta_minus_mu.transpose()).getEntry(0,0);

            double sd0_square = 1 / rngG.nextDouble(a, b[i]);
            meansd[i] = Math.sqrt(sd0_square);
            for(int j = 0; j < this.P; j++)  this.sd0[i][j] = meansd[i];
        }
        if(verbose){
            System.out.printf("Resampled SD0 "+ Arrays.toString(meansd) + "\n");
        }

    }

    public void resample_tau(int[] binary_indices, int[] cont_indices, COV_model model){
        if(cont_indices.length < 1) return;

        double alpha = 2.0;
        double beta = 1.0;

        double meannewtau = 0;

        for(int j = 0; j < cont_indices.length; j++){
            RealMatrix tildeR11 = model.corr_by_tau.getSubMatrix(this.left_index[cont_indices[j]],
                    this.left_index[cont_indices[j]]);
            RealMatrix gR12 = model.corr_by_tau.getSubMatrix(this.left_index[cont_indices[j]],
                    this.remove_index[cont_indices[j]]);
            for(int ii = 0; ii < P-1; ii++) gR12.setEntry(ii, 0, gR12.getEntry(ii, 0) / model.tau[cont_indices[j]]);

            double R22 = model.corr.getEntry(cont_indices[j], cont_indices[j]);
            RealMatrix R12tR11inv = new LUDecomposition(tildeR11).getSolver().getInverse().preMultiply(gR12
                    .transpose());
            double schur = R22 - gR12.preMultiply(R12tR11inv).getEntry(0,0);
            double[] b = new double[this.N];
            for(int ii = 0; ii < this.N; ii++) {
                int g = ii < this.N_test ? this.membership_test[ii] : this.membership[ii];
                int index = 0;
                for (int jj = 0; jj < this.P; jj++) {
                    if(jj != j){
                        b[ii] += R12tR11inv.getEntry(0, index) * (this.latent[ii][jj] - this.Delta[g][jj]);
                        index ++;
                    }
                }
            }
            double[] mu_new = new double[1];
            double[][] var_new = new double[1][1];
            for(int ii = 0; ii < this.N; ii++) {
                int g = ii < this.N_test ? this.membership_test[ii] : this.membership[ii];
                mu_new[0] += b[ii] * (this.latent[ii][j] - this.Delta[g][j]);
                var_new[0][0] += Math.pow(this.latent[ii][j] - this.Delta[g][j], 2);
            }
            mu_new[0] /= var_new[0][0];
            var_new[0][0] = schur / var_new[0][0];

            double[] current = new double[1];
            current[0] = 1/model.tau[j];
            double[] Nminus2 = new double[1];
            Nminus2[0] = this.N - 2;
            Random rand = new Random();
            double[] invsd = ESSsampler.sample(current, 4, mu_new, var_new, true, Nminus2, gR12, gR12, gR12, gR12,
                    rand, 1000);
            model.tau[j] = 1/invsd[0];
            meannewtau += model.tau[j];
        }
        System.out.println("New sampled marginal sd for continuous rv:  " + meannewtau / cont_indices.length);
    }


    public static void main(String[] args){
        int seed = 7;
        Random rand = new Random(seed);
        NormalDistribution rngN = new NormalDistribution();
        DoubleMersenneTwister rngEngine = new DoubleMersenneTwister(seed);
        Exponential rngE = new Exponential(1, rngEngine);
        double mean = 0;
        double sd = 1;
        double min = Double.MAX_VALUE * (-1);
        double max = -0.19;

        for(int i = 0; i < 1000; i++) {
            System.out.printf("%.8f, ", MathUtil.truncNormal(rand, rngN, rngE, mean, sd, min, max, 5));
        }
        System.out.println();
        for(int i = 0; i < 1000; i++) {
            System.out.printf("%.8f, ", MathUtil.leftTruncNormal(mean, sd, 0.19, rand, rngE));
        }


    }



}
