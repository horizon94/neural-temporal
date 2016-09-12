package org.chboston.cnlp.temporal.neural;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.StringUtils;
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
      List<String> outcomes = new ArrayList<>();
      // label for start token:
      outcomes.add("O");
      if (this.isTraining()) {
        List<TimeMention> times = JCasUtil.selectCovered(jcas, TimeMention.class, sentence);
        outcomes.addAll(this.timeChunking.createOutcomes(jcas, tokens, times));
      }
      // during prediction, the list of outcomes predicted so far
      else {
        outcomes = new ArrayList<String>();
      }
      
      // First write an instance for each token -- here just a label and the token (label at training time only)
      List<Feature> tokenFeats= new ArrayList<>();
      tokenFeats.add(new Feature("START"));
      
      for (BaseToken token : tokens) {
        tokenFeats.add(new Feature(token.getCoveredText().toLowerCase()));        
      }
      
      // At the end of sentence, write a dummy instance to indicate EOS, at test time
      // compile the labels into chunks and build time expressions:
      tokenFeats.add(new Feature("EOS"));
      outcomes.add("O");
      if(!this.isTraining()){
        String labels = this.classifier.classify(tokenFeats);
        this.timeChunking.createChunks(jcas, tokens, Arrays.asList(labels.split(" ")).subList(1, tokenFeats.size()-1));
      }else{
        this.dataWriter.write(new Instance<>(StringUtils.join(outcomes, " "), tokenFeats));
      }
    }
  }

}
