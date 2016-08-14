package org.chboston.cnlp.temporal.neural;

import java.util.ArrayList;
import java.util.List;

import org.apache.ctakes.core.util.DocumentIDAnnotationUtil;
import org.apache.ctakes.neural.feature.TokensSequenceWithWindowExtractor;
import org.apache.ctakes.temporal.ae.TemporalEntityAnnotator_ImplBase;
import org.apache.ctakes.typesystem.type.syntax.BaseToken;
import org.apache.ctakes.typesystem.type.textsem.TimeMention;
import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.uima.UIMAFramework;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.Level;
import org.apache.uima.util.Logger;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;
import org.cleartk.ml.chunking.BioChunking;

public class RnnTimexAnnotator extends TemporalEntityAnnotator_ImplBase {
  Logger logger = UIMAFramework.getLogger(RnnTimexAnnotator.class);
  private final static int WINDOW_SIZE = 10;
  private TokensSequenceWithWindowExtractor seqExtractor = new TokensSequenceWithWindowExtractor(WINDOW_SIZE);

  private BioChunking<BaseToken, TimeMention> timeChunking;

  @Override
  public void initialize(UimaContext context)
      throws ResourceInitializationException {
    super.initialize(context);
    
    this.timeChunking = new BioChunking<BaseToken, TimeMention>(BaseToken.class, TimeMention.class);

  }
  
  @Override
  public void process(JCas jcas, Segment segment) throws AnalysisEngineProcessException {
    String documentId = DocumentIDAnnotationUtil.getDocumentID(jcas);
    if(documentId != null){
      logger.log(Level.INFO, "Processing document " + documentId);
    }

    // classify tokens within each sentence
    for (Sentence sentence : JCasUtil.selectCovered(jcas, Sentence.class, segment)) {
      List<BaseToken> tokens = JCasUtil.selectCovered(jcas, BaseToken.class, sentence);
      
      // during training, the list of all outcomes for the tokens
      List<String> outcomes;
      if (this.isTraining()) {
        List<TimeMention> times = JCasUtil.selectCovered(jcas, TimeMention.class, sentence);
        outcomes = this.timeChunking.createOutcomes(jcas, tokens, times);
      }
      // during prediction, the list of outcomes predicted so far
      else {
        outcomes = new ArrayList<String>();
      }
      
      // First write an instance for each token -- here just a label and the token (label at training time only)
      int tokenIndex = 0;
      for (BaseToken token : tokens) {
        List<Feature> tokenFeats= new ArrayList<>();
        tokenFeats.add(new Feature(token.getCoveredText().toLowerCase()));
//        tokenFeats.add(new Feature(String.format("%s", tokenIndex==0 ? "O" : outcomes.get(tokenIndex-1))));
        
        if(this.isTraining()){
          String outcome = outcomes.get(tokenIndex);
          this.dataWriter.write(new Instance<>(outcome, tokenFeats));
        }else{
          outcomes.add(this.classifier.classify(tokenFeats));
        }
        tokenIndex++;
      }
      
      // At the end of sentence, write a dummy instance to indicate EOS, at test time
      // compile the labels into chunks and build time expressions:
      List<Feature> tokenFeats = new ArrayList<>();
      tokenFeats.add(new Feature("EOS"));
      if(!this.isTraining()){
        this.timeChunking.createChunks(jcas, tokens, outcomes);
        this.classifier.classify(tokenFeats);
      }else{
        this.dataWriter.write(new Instance<>("O", tokenFeats));
      }
    }
  }

}
