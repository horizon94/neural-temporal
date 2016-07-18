package org.apache.ctakes.temporal.ae;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.ctakes.temporal.ae.TemporalRelationExtractorAnnotator.IdentifiedAnnotationPair;
import org.apache.ctakes.temporal.nn.eventTimeRelationPrinter;
import org.apache.ctakes.typesystem.type.relation.BinaryTextRelation;
import org.apache.ctakes.typesystem.type.relation.RelationArgument;
import org.apache.ctakes.typesystem.type.relation.TemporalTextRelation;
import org.apache.ctakes.typesystem.type.textsem.EventMention;
import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;
import org.apache.ctakes.typesystem.type.textsem.TimeMention;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;
import org.cleartk.util.ViewUriUtil;

import com.google.common.collect.Lists;

public class EventTimeCNNAnnotator extends CleartkAnnotator<String> {

	public static final String NO_RELATION_CATEGORY = "none";//-NONE-

	public EventTimeCNNAnnotator() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public void process(JCas jCas) throws AnalysisEngineProcessException {
		//get all gold relation lookup
		Map<List<Annotation>, BinaryTextRelation> relationLookup;
		relationLookup = new HashMap<>();
		if (this.isTraining()) {
			relationLookup = new HashMap<>();
			for (BinaryTextRelation relation : JCasUtil.select(jCas, BinaryTextRelation.class)) {
				Annotation arg1 = relation.getArg1().getArgument();
				Annotation arg2 = relation.getArg2().getArgument();
				// The key is a list of args so we can do bi-directional lookup
				List<Annotation> key = Arrays.asList(arg1, arg2);
				if(relationLookup.containsKey(key)){
					String reln = relationLookup.get(key).getCategory();
					System.err.println("Error in: "+ ViewUriUtil.getURI(jCas).toString());
					System.err.println("Error! This attempted relation " + relation.getCategory() + " already has a relation " + reln + " at this span: " + arg1.getCoveredText() + " -- " + arg2.getCoveredText());
				}else{
					relationLookup.put(key, relation);
				}
			}
		}

		// go over sentences, extracting event-time relation instances
		for(Sentence sentence : JCasUtil.select(jCas, Sentence.class)) {
			// collect all relevant relation arguments from the sentence
			List<IdentifiedAnnotationPair> candidatePairs =
					getCandidateRelationArgumentPairs(jCas, sentence);

			// walk through the pairs of annotations
			for (IdentifiedAnnotationPair pair : candidatePairs) {
				IdentifiedAnnotation arg1 = pair.getArg1();
				IdentifiedAnnotation arg2 = pair.getArg2();

				String context;
				if(arg2.getBegin() < arg1.getBegin()) {
					// ... time ... event ... scenario
					context = eventTimeRelationPrinter.getTokensBetween(jCas, sentence, arg2, "t", arg1, "e", 2);  
				} else {
					// ... event ... time ... scenario
					context = eventTimeRelationPrinter.getTokensBetween(jCas, sentence, arg1, "e", arg2, "t", 2);
				}

				//derive features based on context:
				List<Feature> feats = new ArrayList<>();
				String[] tokens = context.split(" ");
				for (String token: tokens){
					feats.add(new Feature(token.toLowerCase()));
				}

				// during training, feed the features to the data writer
				if (this.isTraining()) {
					String category = getRelationCategory(relationLookup, arg1, arg2);
					if (category == null) {
						category = NO_RELATION_CATEGORY.toLowerCase();
					}else{
						category = category.toLowerCase();
					}
					this.dataWriter.write(new Instance<>(category, feats));
				}

				// during classification feed the features to the classifier and create annotations
				else {
					String predictedCategory = this.classifier.classify(feats);

					// add a relation annotation if a true relation was predicted
					if (predictedCategory != null && !predictedCategory.equals(NO_RELATION_CATEGORY.toLowerCase())) {

						// if we predict an inverted relation, reverse the order of the arguments
						if (predictedCategory.endsWith("-1")) {
							predictedCategory = predictedCategory.substring(0, predictedCategory.length() - 2);
							if(arg1 instanceof TimeMention){
								IdentifiedAnnotation temp = arg1;
								arg1 = arg2;
								arg2 = temp;
							}
						}else{
							if(arg1 instanceof EventMention){
								IdentifiedAnnotation temp = arg1;
								arg1 = arg2;
								arg2 = temp;
							}
						}

						createRelation(jCas, arg1, arg2, predictedCategory.toUpperCase(), 0.0);
					}
				}
			}

		}
	}

	/**
	 * original way of getting label
	 * @param relationLookup
	 * @param arg1
	 * @param arg2
	 * @return
	 */
	//	protected String getRelationCategory(
	//			Map<List<Annotation>, BinaryTextRelation> relationLookup,
	//			IdentifiedAnnotation arg1,
	//			IdentifiedAnnotation arg2) {
	//		BinaryTextRelation relation = relationLookup.get(Arrays.asList(arg1, arg2));
	//		String category = null;
	//		if (relation != null) {
	//			category = relation.getCategory();
	//		} else {
	//			relation = relationLookup.get(Arrays.asList(arg2, arg1));
	//			if (relation != null) {
	//				if(relation.getCategory().equals("OVERLAP")){
	//					category = relation.getCategory();
	//				}else{
	//					category = relation.getCategory() + "-1";
	//				}
	//			}
	//		}
	//
	//		return category;
	//	}

	/** Dima's way of getting lables
	 * @param relationLookup
	 * @param arg1
	 * @param arg2
	 * @return
	 */
	protected String getRelationCategory(Map<List<Annotation>, BinaryTextRelation> relationLookup,
			IdentifiedAnnotation arg1,
			IdentifiedAnnotation arg2){
		BinaryTextRelation relation = relationLookup.get(Arrays.asList(arg1, arg2));
		String category = null;
		if (relation != null) {
			category = relation.getCategory();
			if(arg1 instanceof EventMention){
				category = category + "-1";
			}
		} else {
			relation = relationLookup.get(Arrays.asList(arg2, arg1));
			if (relation != null) {
				category = relation.getCategory();
				if(arg2 instanceof EventMention){
					category = category + "-1";
				}
			}
		}

		return category;

	}


	protected void createRelation(JCas jCas, IdentifiedAnnotation arg1,
			IdentifiedAnnotation arg2, String predictedCategory, double confidence) {
		RelationArgument relArg1 = new RelationArgument(jCas);
		relArg1.setArgument(arg1);
		relArg1.setRole("Arg1");
		relArg1.addToIndexes();
		RelationArgument relArg2 = new RelationArgument(jCas);
		relArg2.setArgument(arg2);
		relArg2.setRole("Arg2");
		relArg2.addToIndexes();
		TemporalTextRelation relation = new TemporalTextRelation(jCas);
		relation.setArg1(relArg1);
		relation.setArg2(relArg2);
		relation.setCategory(predictedCategory);
		relation.setConfidence(confidence);
		relation.addToIndexes();
	}

	private List<IdentifiedAnnotationPair> getCandidateRelationArgumentPairs(JCas jCas, Sentence sentence) {
		//		Map<EventMention, Collection<EventMention>> coveringMap =
		//				JCasUtil.indexCovering(jCas, EventMention.class, EventMention.class);

		List<IdentifiedAnnotationPair> pairs = Lists.newArrayList();
		for (EventMention event : JCasUtil.selectCovered(jCas, EventMention.class, sentence)) {
			boolean eventValid = false;
			if (event.getClass().equals(EventMention.class)) {//event is a gold event
				eventValid = true;
			}

			if(eventValid){
				// ignore subclasses like Procedure and Disease/Disorder
				if(this.isTraining()){//if training mode, train on both gold event and span-overlapping system events
					for (TimeMention time : JCasUtil.selectCovered(jCas, TimeMention.class, sentence)) {

						//						Collection<EventMention> eventList = coveringMap.get(event);
						//						for(EventMention covEvent : eventList){
						//							pairs.add(new IdentifiedAnnotationPair(covEvent, time));
						//						}
						pairs.add(new IdentifiedAnnotationPair(event, time));
					}
				}else{//if testing mode, only test on system generated events
					for (TimeMention time : JCasUtil.selectCovered(jCas, TimeMention.class, sentence)) {
						pairs.add(new IdentifiedAnnotationPair(event, time));
					}
				}
			}
		}

		return pairs;
	}

}
