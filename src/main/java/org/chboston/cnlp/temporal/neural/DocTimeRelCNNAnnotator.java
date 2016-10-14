/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.chboston.cnlp.temporal.neural;

import java.io.File;
import java.util.ArrayList;
//import java.io.IOException;
import java.util.List;
//import java.util.Map;

import org.apache.ctakes.neural.feature.TokensSequenceWithWindowExtractor;
import org.apache.ctakes.typesystem.type.refsem.Event;
import org.apache.ctakes.typesystem.type.refsem.EventProperties;
import org.apache.ctakes.typesystem.type.textsem.EventMention;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;
import org.cleartk.ml.jar.GenericJarClassifierFactory;

public class DocTimeRelCNNAnnotator extends CleartkAnnotator<String> {

	public static AnalysisEngineDescription createAnnotatorDescription(String modelPath)
			throws ResourceInitializationException {
		return AnalysisEngineFactory.createEngineDescription(
				DocTimeRelCNNAnnotator.class,
				CleartkAnnotator.PARAM_IS_TRAINING,
				false,
				GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
				modelPath);
	}

	private final static int WINDOW_SIZE = 40;
	private TokensSequenceWithWindowExtractor seqExtractor = new TokensSequenceWithWindowExtractor(WINDOW_SIZE);

	@Override
	public void process(JCas jCas) throws AnalysisEngineProcessException {
		for (EventMention eventMention : JCasUtil.select(jCas, EventMention.class)) {
			List<Sentence> sents = JCasUtil.selectCovering(jCas, Sentence.class, eventMention);
			List<Feature> features = new ArrayList<>();
			features.addAll(this.seqExtractor.extract(jCas, eventMention));
			if (this.isTraining()) {
				if(eventMention.getEvent() != null){
					String outcome = eventMention.getEvent().getProperties().getDocTimeRel();
					this.dataWriter.write(new Instance<>(outcome, features));
				}
			} else {
				String outcome = this.classifier.classify(features);
				if(eventMention.getEvent()==null){
					Event event = new Event(jCas);
					EventProperties props = new EventProperties(jCas);
					props.setDocTimeRel(outcome);
					event.setProperties(props);
					eventMention.setEvent(event);
				}else{
					eventMention.getEvent().getProperties().setDocTimeRel(outcome);
				}			
			}
		}
	}
}
