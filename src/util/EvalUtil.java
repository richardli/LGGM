package util;

import cern.colt.Arrays;
import org.apache.commons.math3.linear.RealMatrix;
import sampler.COV_model;
import sampler.Latent_model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by zehangli on 10/26/16.
 */
public class EvalUtil {

    public static double[] getnorm(RealMatrix mat1, RealMatrix mat2){
        RealMatrix diff = mat1.subtract(mat2);
        int P = mat1.getRowDimension();

        double fnorm = diff.getFrobeniusNorm();
        double inftynorm = diff.getNorm();
        double mnorm = 0;
        for(int j = 0; j < P; j++) {
            for (int jj = 0; jj < P; jj++) {
                if(diff.getEntry(j, jj) > mnorm) mnorm = diff.getEntry(j, jj);
            }
        }

        double[] out = new double[3];
        out[0] = fnorm;
        out[1] = inftynorm;
        out[2] = mnorm;
        return(out);
    }
    public static double getAUC(int[][] mat1, RealMatrix truth){
        double AUC = 0;


        return(AUC);
    }

    // mat1 is the estimator, mat2 is the truth
    public static int[] getclassification(int[][] mat1, RealMatrix truth){
        int P = truth.getRowDimension();
        int[] out = new int[4];
        for(int i = 0; i < P; i++){
            for(int j = 0; j < P; j++){
                if(i == j) continue;
                // TP
                out[0] += (mat1[i][j] != 0 & truth.getEntry(i, j) != 0) ? 1 : 0;
                // FP
                out[1] += (mat1[i][j] != 0 & truth.getEntry(i, j) == 0) ? 1 : 0;
                // TN
                out[2] += (mat1[i][j] == 0 & truth.getEntry(i, j) == 0) ? 1 : 0;
                // FN
                out[3] += (mat1[i][j] == 0 & truth.getEntry(i, j) != 0) ? 1 : 0;
            }
        }
        return(out);
    }


    public static void save_draw_px(Latent_model model, int nitr, int G){
        for(int j = 0; j < model.P; j++){
            for(int g = 0; g < G; g++){
                model.mean_draw[g][nitr][j] = model.data.Delta[g][j];
            }
            model.sd_draw[nitr][j] = model.data.d[j];
            for(int jj = 0; jj < model.P; jj++){
                model.prec_draw[nitr][j][jj] = model.cov_px.invcorr.getEntry(j, jj);
                model.corr_draw[nitr][j][jj] = model.cov_px.corr_by_tau.getEntry(j, jj);
            }
        }
    }


    public static void save_draw_sssl(Latent_model model, int nitr, int G){
        for(int j = 0; j < model.P; j++){
            for(int g = 0; g < G; g++){
                model.mean_draw[g][nitr][j] = model.data.Delta[g][j];
            }
            model.sd_draw[nitr][j] = model.data.d[j];
            for(int jj = 0; jj < model.P; jj++){
                model.prec_draw[nitr][j][jj] = model.cov_sssl.invcorr.getEntry(j, jj);
                model.corr_draw[nitr][j][jj] = model.cov_sssl.corr_by_tau.getEntry(j, jj);
            }
        }
    }


    public static void save_draw_glasso(Latent_model model, int nitr, int G){
        for(int j = 0; j < model.P; j++){
            for(int g = 0; g < G; g++){
                model.mean_draw[g][nitr][j] = model.data.Delta[g][j];
            }
            model.sd_draw[nitr][j] = model.data.d[j];
            for(int jj = 0; jj < model.P; jj++){
                model.prec_draw[nitr][j][jj] = model.cov_glasso.invcorr.getEntry(j, jj);
                model.corr_draw[nitr][j][jj] = model.cov_glasso.corr_by_tau.getEntry(j, jj);
            }
        }
    }

    public static void savearray(ArrayList array, String directory, String pre, String covType, String post) throws
            IOException {
        String file = directory + pre + "_" + post + ".txt";

        File theDir = new File(directory);

        // if the directory does not exist, create it
        if (!theDir.exists()) {
            System.out.println("\ncreating directory: " + directory);
            boolean result = false;
            try {
                theDir.mkdir();
                result = true;
            } catch (SecurityException se) {
                System.out.println("Cannot create directory");
            }
            if (result) {
                System.out.println("DIR created");
            }
        }

        BufferedWriter bw = new BufferedWriter(new FileWriter(file));
        bw.write(Arrays.toString(array.toArray()).replace("[", "").replace("]", "\n"));
        bw.close();
    }
    public static void save(Latent_model model, COV_model cov_model, String directory, String pre, String covType,
                            boolean
            update_group) throws
            IOException {
        int G = model.G;
        String file_corr = directory + pre + "_corr_out_mean.txt";
        String file_corr2 = directory + pre + "_corr_out_core_mean.txt";
        String file_prec = directory + pre + "_invcorr_out_mean.txt";
        String file_mean = directory + pre + "_mean_out_mean.txt";

        File theDir = new File(directory);

        // if the directory does not exist, create it
        if (!theDir.exists()) {
            System.out.println("\ncreating directory: " + directory);
            boolean result = false;
            try{
                theDir.mkdir();
                result = true;
            }
            catch(SecurityException se){
                System.out.println("Cannot create directory");
            }
            if(result) {
                System.out.println("DIR created");
            }
        }

        BufferedWriter bw_prec = new BufferedWriter(new FileWriter(file_prec));
        BufferedWriter bw_corr = new BufferedWriter(new FileWriter(file_corr));
        BufferedWriter bw_corr2 = new BufferedWriter(new FileWriter(file_corr2));
        BufferedWriter bw_mean = new BufferedWriter(new FileWriter(file_mean));

        int P = model.prec_draw[0].length;

        double NitrNow = model.corr_draw.length - 1.0;
        double[][] mean_mean = new double[G][P];
        double[][] corr_mean = new double[P][P];
        double[][] prec_mean = new double[P][P];
        // remove first
        for(int nitr = 1; nitr < model.corr_draw.length; nitr ++) {
            for (int g = 0; g < G; g++) {
                for (int j = 0; j < P; j++) {
                    mean_mean[g][j] += model.mean_draw[g][nitr][j] / NitrNow;
                }
            }
        }
        for (int g = 0; g < G; g++) {
            bw_mean.write(Arrays.toString(mean_mean[g]).replace("[", "").replace("]", "\n"));
        }

        // remove first
        for(int nitr = 1; nitr < model.corr_draw.length; nitr ++) {
            for (int j = 0; j < P; j++) {
                for (int jj = 0; jj < P; jj++) {
                    corr_mean[j][jj] += model.corr_draw[nitr][j][jj] / NitrNow;
                    prec_mean[j][jj] += model.prec_draw[nitr][j][jj] / NitrNow;
                }
            }
        }
        for(int j = 0; j < model.P; j++){
            bw_prec.write(Arrays.toString(prec_mean[j]).replace("[", "").replace("]", "\n"));
            bw_corr.write(Arrays.toString(corr_mean[j]).replace("[", "").replace("]", "\n"));
            bw_corr2.write(Arrays.toString(cov_model.corr_ave.getColumn(j)).replace("[", "").replace("]", "\n"));
        }

        bw_prec.close();
        bw_corr.close();
        bw_corr2.close();
        bw_mean.close();

        if(covType.equals("SSSL")){
            String file_inclusion = directory + pre + "_inclusion_out.txt";
            BufferedWriter bw_inclusion = new BufferedWriter(new FileWriter(file_inclusion));
            for(int j = 0; j < model.P; j++){
                bw_inclusion.write(Arrays.toString(model.cov_sssl.inclusion_ave[j]).replace("[", "").replace("]", "\n"));
            }
            bw_inclusion.close();
        }
        if(covType.equals("GLASSO")){
            String file_inclusion = directory + pre + "_inclusion_out.txt";
            BufferedWriter bw_inclusion = new BufferedWriter(new FileWriter(file_inclusion));
            for(int j = 0; j < model.P; j++){
                bw_inclusion.write(Arrays.toString(model.cov_glasso.inclusion_ave[j]).replace("[", "").replace("]", "\n"));
            }
            bw_inclusion.close();
        }

        if(update_group){
            if(model.true_prob != null){
                String file_prob_true = directory + pre + "_prob_true.txt";
                BufferedWriter bw_prob_true = new BufferedWriter(new FileWriter(file_prob_true));
                bw_prob_true.write(Arrays.toString(model.true_prob).replace("[", "").replace("]", "\n"));
                bw_prob_true.close();
            }

            String file_prob = directory + pre + "_prob_out.txt";
            BufferedWriter bw_prob = new BufferedWriter(new FileWriter(file_prob));
            for(int j = 0; j < model.post_prob_pop.length; j++){
                    bw_prob.write(Arrays.toString(model.post_prob_pop[j]).replace("[", "").replace("]", "\n"));
            }
            bw_prob.close();

            double[][] prob_mean = new double[model.post_prob_draw.length][model.post_prob_draw[0][0].length];
            NitrNow = model.post_prob_draw[0].length - 1.0;
            for(int i = 0; i < model.post_prob_draw.length; i++){
                // remove the first one
                for(int j = 1; j < model.post_prob_draw[0].length; j++){
                    for(int k = 0; k < model.post_prob_draw[0][0].length; k++){
                        prob_mean[i][k] += model.post_prob_draw[i][j][k] / NitrNow;
                    }
                }
            }
            String file_assignment = directory + pre + "_assignment_out_mean.txt";
            BufferedWriter bw_inclusion = new BufferedWriter(new FileWriter(file_assignment));
            for(int j = 0; j < prob_mean.length; j++){
                    bw_inclusion.write(Arrays.toString(prob_mean[j]).replace("[", "").replace("]", "\n"));
            }
            bw_inclusion.close();
        }

    }


    public static void save_full(Latent_model model, String directory, String pre, String covType, boolean
            update_group) throws
            IOException {
        int G = model.G;
        String file_corr = directory + pre + "_corr_out.txt";
        String file_prec = directory + pre + "_prec_out.txt";
        String file_mean = directory + pre + "_mean_out.txt";

        File theDir = new File(directory);

// if the directory does not exist, create it
        if (!theDir.exists()) {
            System.out.println("\ncreating directory: " + directory);
            boolean result = false;
            try{
                theDir.mkdir();
                result = true;
            }
            catch(SecurityException se){
                System.out.println("Cannot create directory");
            }
            if(result) {
                System.out.println("DIR created");
            }
        }

        BufferedWriter bw_prec = new BufferedWriter(new FileWriter(file_prec));
        BufferedWriter bw_corr = new BufferedWriter(new FileWriter(file_corr));
        BufferedWriter bw_mean = new BufferedWriter(new FileWriter(file_mean));

        for(int nitr = 0; nitr < model.corr_draw.length; nitr ++){
            for(int g = 0; g < G; g++){
                bw_mean.write(Arrays.toString(model.mean_draw[g][nitr]).replace("[", "").replace("]", "\n"));
            }
            for(int j = 0; j < model.P; j++){
                bw_prec.write(Arrays.toString(model.prec_draw[nitr][j]).replace("[", "").replace("]", "\n"));
                bw_corr.write(Arrays.toString(model.corr_draw[nitr][j]).replace("[", "").replace("]", "\n"));
            }
        }
        bw_prec.close();
        bw_corr.close();
        bw_mean.close();


//        if(covType.equals("SSSL")){
//            String file_inclusion = directory + pre + "_inclusion_out.txt";
//            BufferedWriter bw_inclusion = new BufferedWriter(new FileWriter(file_inclusion));
//            for(int j = 0; j < model.P; j++){
//                bw_inclusion.write(Arrays.toString(model.cov_sssl.inclusion_ave[j]).replace("[", "").replace("]", "\n"));
//            }
//            bw_inclusion.close();
//        }
        if(covType.equals("GLASSO")){
            String file_inclusion = directory + pre + "_inclusion_out.txt";
            BufferedWriter bw_inclusion = new BufferedWriter(new FileWriter(file_inclusion));
            for(int j = 0; j < model.P; j++){
                bw_inclusion.write(Arrays.toString(model.cov_glasso.inclusion_ave[j]).replace("[", "").replace("]", "\n"));
            }
            bw_inclusion.close();
        }

        if(update_group){
            if(model.true_prob != null){
                String file_prob_true = directory + pre + "_prob_true.txt";
                BufferedWriter bw_prob_true = new BufferedWriter(new FileWriter(file_prob_true));
                bw_prob_true.write(Arrays.toString(model.true_prob).replace("[", "").replace("]", "\n"));
                bw_prob_true.close();
            }

            String file_prob = directory + pre + "_prob_out.txt";
            BufferedWriter bw_prob = new BufferedWriter(new FileWriter(file_prob));
            for(int j = 0; j < model.post_prob_pop.length; j++){
                bw_prob.write(Arrays.toString(model.post_prob_pop[j]).replace("[", "").replace("]", "\n"));
            }
            bw_prob.close();

            String file_assignment = directory + pre + "_assignment_out.txt";
            BufferedWriter bw_inclusion = new BufferedWriter(new FileWriter(file_assignment));
            for(int j = 0; j < model.post_prob_draw.length; j++){
                for(int k = 0; k < model.post_prob_draw[j].length; k++){
                    bw_inclusion.write(Arrays.toString(model.post_prob_draw[j][k]).replace("[", "").replace("]", "\n"));
                }
            }
            bw_inclusion.close();
        }

    }

    public static void savetruth(Latent_model model, String directory, String pre, String covType, double[][] prec_true, double[][] corr_true, double[][] mean_true) throws
            IOException {
        String file_X = directory + pre + "_X.txt";
        String file_prec = directory + pre + "_prec_out_truth.txt";
        String file_corr = directory + pre + "_corr_out_truth.txt";
        String file_mean = directory + pre + "_mean_out_truth.txt";

        File theDir = new File(directory);
        // if the directory does not exist, create it
        if (!theDir.exists()) {
            System.out.println("\ncreating directory: " + directory);
            boolean result = false;
            try{
                theDir.mkdir();
                result = true;
            }
            catch(SecurityException se){
                System.out.println("Cannot create directory");
            }
            if(result) {
                System.out.println("DIR created");
            }
        }
        BufferedWriter bw_X = new BufferedWriter(new FileWriter(file_X));
        BufferedWriter bw_corr = new BufferedWriter(new FileWriter(file_corr));
        BufferedWriter bw_prec = new BufferedWriter(new FileWriter(file_prec));
        BufferedWriter bw_mean = new BufferedWriter(new FileWriter(file_mean));
        for(int j = 0; j < mean_true.length; j++) {
            bw_mean.write(Arrays.toString(mean_true[j]).replace("[", "").replace("]", "\n"));
        }
        for(int j = 0; j < model.P; j++){
            bw_prec.write(Arrays.toString(prec_true[j]).replace("[", "").replace("]", "\n"));
        }
        for(int j = 0; j < model.P; j++){
            bw_corr.write(Arrays.toString(corr_true[j]).replace("[", "").replace("]", "\n"));
        }
        for(int j = 0; j < model.data.N; j++){
            bw_X.write(model.data.membership[j] + ", ");
            bw_X.write(Arrays.toString(model.data.data[j]).replace("[", "").replace("]", "\n"));
        }

        bw_corr.close();
        bw_prec.close();
        bw_mean.close();
        bw_X.close();


    }

}
