package sampler;

/**
 * Created by zehangli on 8/11/17.
 */
public class Latent_classifier_multiple extends Latent_classifier {

    public Latent_classifier_multiple(int Nitr, int burnin, int thin, int N, int P, int G, String covType){
        super(Nitr, burnin, thin, N, P, G, covType);
    }
    public Latent_classifier_multiple(int Nitr, int N, int P, int G, String covType){
        super(Nitr, N, 0, P, G, covType);
    }
    public Latent_classifier_multiple(int Nitr, int N, int N_test, int P, int G, String covType){super(Nitr, N, N_test, P, G, covType);}

}
