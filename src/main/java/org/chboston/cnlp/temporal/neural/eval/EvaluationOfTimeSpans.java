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
package org.chboston.cnlp.temporal.neural.eval;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.logging.Level;

import org.apache.ctakes.neural.ScriptStringFeatureDataWriter;
import org.apache.ctakes.temporal.eval.EvaluationOfAnnotationSpans_ImplBase;
import org.apache.ctakes.temporal.eval.Evaluation_ImplBase;
import org.apache.ctakes.temporal.eval.I2B2Data;
import org.apache.ctakes.temporal.eval.THYMEData;
import org.apache.ctakes.typesystem.type.textsem.TimeMention;
import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.apache.uima.resource.ResourceInitializationException;
import org.chboston.cnlp.temporal.neural.RnnTimexAnnotator;
import org.cleartk.eval.AnnotationStatistics;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.jar.DefaultDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.cleartk.ml.jar.JarClassifierBuilder;

import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

public class EvaluationOfTimeSpans extends EvaluationOfAnnotationSpans_ImplBase {

	static interface Options extends Evaluation_ImplBase.Options {		
		@Option(defaultValue="target/eval/time-spans")
		public File getBaseDirectory();
	}

	public static void main(String[] args) throws Exception {
		Options options = CliFactory.parseArguments(Options.class, args);
		List<Integer> trainItems = null;
    List<Integer> devItems = null;
    List<Integer> testItems = null;
		
    List<Integer> patientSets = options.getPatients().getList();
    if(options.getXMLFormat() == XMLFormat.I2B2){
      trainItems = I2B2Data.getTrainPatientSets(options.getXMLDirectory());
      devItems = I2B2Data.getDevPatientSets(options.getXMLDirectory());
      testItems = I2B2Data.getTestPatientSets(options.getXMLDirectory());
    }else{
      trainItems = THYMEData.getPatientSets(patientSets, options.getTrainRemainders().getList());
      devItems = THYMEData.getPatientSets(patientSets, options.getDevRemainders().getList());
      testItems = THYMEData.getPatientSets(patientSets, options.getTestRemainders().getList());
    }
		
		List<Integer> allTrain = new ArrayList<>(trainItems);
		List<Integer> allTest = null;
		
		if(options.getTest()){
		  allTrain.addAll(devItems);
		  allTest = new ArrayList<>(testItems);
		}else{
		  allTest = new ArrayList<>(devItems);
		}
		
		// run one evaluation per annotator class
		EvaluationOfTimeSpans evaluation = new EvaluationOfTimeSpans(
		    options.getBaseDirectory(),
		    options.getRawTextDirectory(),
		    options.getXMLDirectory(),
		    options.getXMLFormat(),
		    options.getSubcorpus(),
		    options.getXMIDirectory(),
		    options.getTreebankDirectory(),
		    options.getPrintOverlappingSpans());
		evaluation.prepareXMIsFor(patientSets);
		evaluation.skipTrain = options.getSkipTrain();
		evaluation.printErrors = options.getPrintErrors();
		evaluation.kernelParams = new String[]{options.getKernelParams()};
		if(options.getI2B2Output()!=null) evaluation.setI2B2Output(options.getI2B2Output() + "/RnnTimexAnnotator");
		String name = String.format("RnnTimexAnnotator.errors");
		evaluation.setLogging(Level.FINE, new File("target/eval", name));
		AnnotationStatistics<String> stats = evaluation.trainAndTest(allTrain, allTest);

		// print out models, ordered by F1
		System.err.println(stats);
	}

	private boolean skipTrain = false;
	
	public EvaluationOfTimeSpans(
			File baseDirectory,
			File rawTextDirectory,
			File xmlDirectory,
			XMLFormat xmlFormat,
			Subcorpus subcorpus,
			File xmiDirectory,
			File treebankDirectory,
			boolean printOverlapping) {
		super(baseDirectory, rawTextDirectory, xmlDirectory, xmlFormat, subcorpus, xmiDirectory, treebankDirectory, TimeMention.class);
		this.printOverlapping = printOverlapping;
	}
	
	@Override
	public void train(CollectionReader reader, File directory) throws Exception{
	  if(!skipTrain){
	    super.train(reader, directory);
	  }
	}
	
	@Override
	protected AnalysisEngineDescription getDataWriterDescription(File directory)
			throws ResourceInitializationException {
	  return AnalysisEngineFactory.createEngineDescription(RnnTimexAnnotator.class,
	      RnnTimexAnnotator.PARAM_IS_TRAINING,
	      true,
	      DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
	      ScriptStringFeatureDataWriter.class,
	      DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
	      this.getModelDirectory(directory),
	      ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
        "scripts/keras/timex");
	}

	@Override
	protected void trainAndPackage(File directory) throws Exception {
		JarClassifierBuilder.trainAndPackage(this.getModelDirectory(directory), this.kernelParams);
	}

	@Override
	protected AnalysisEngineDescription getAnnotatorDescription(File directory)
			throws ResourceInitializationException {
	  return AnalysisEngineFactory.createEngineDescription(RnnTimexAnnotator.class,
	      CleartkAnnotator.PARAM_IS_TRAINING,
	      false,
	      GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
	      new File(this.getModelDirectory(directory), "model.jar"));	  
	}

	@Override
	protected Collection<? extends Annotation> getGoldAnnotations(JCas jCas, Segment segment) {
		return selectExact(jCas, TimeMention.class, segment);
	}

	@Override
	protected Collection<? extends Annotation> getSystemAnnotations(JCas jCas, Segment segment) {
		return selectExact(jCas, TimeMention.class, segment);
	}

	private File getModelDirectory(File directory) {
		return new File(directory, "rnn-timex");
	}
}
