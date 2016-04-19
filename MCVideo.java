package burlap.tutorials.video.mc;

import burlap.assignment4.BasicGridWorld;
import burlap.assignment4.util.AnalysisAggregator;
import burlap.assignment4.util.MapPrinter;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.vfa.common.ConcatenatedObjectFeatureVectorGenerator;
import burlap.behavior.singleagent.vfa.fourier.FourierBasis;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.common.GoalBasedRF;
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.singleagent.environment.Environment;
import burlap.oomdp.singleagent.environment.EnvironmentServer;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.HashableStateFactory;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.oomdp.core.states.State;


public class MCVideo {
	private static int MAX_ITERATIONS = 1;
	private static int NUM_INTERVALS = 1;
	
	

	
	private static Policy runValueIteration(MountainCar gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf, boolean showPolicyMap) {
		
		
		System.out.println("//Value Iteration Analysis//");
		
		final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
		ValueIteration vi = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			long startTime = System.nanoTime();
			vi = new ValueIteration(
					domain,
					rf,
					tf,
					0.99,
					hashingFactory,
					-1, numIterations); //Added a very high delta number in order to guarantee that value iteration occurs the max number of iterations
										   //for comparison with the other algorithms.
	
			// run planning from our initial state
			p = vi.planFromState(initialState);
			AnalysisAggregator.addMillisecondsToFinishValueIteration((int) (System.nanoTime()-startTime)/1000000);

			// evaluate the policy with one roll out visualize the trajectory
			ea = p.evaluateBehavior(initialState, rf, tf);
			AnalysisAggregator.addValueIterationReward(calcRewardInEpisode(ea));
			AnalysisAggregator.addStepsToFinishValueIteration(ea.numTimeSteps());
		}
		
//		Visualizer v = gen.getVisualizer();
//		new EpisodeSequenceVisualizer(v, domain, Arrays.asList(ea));
		AnalysisAggregator.printValueIterationResults();
		//MapPrinter.printPolicyMap(vi.getAllStates(), p, gen.getMap());
		System.out.println("\n\n");
		//if(showPolicyMap){
		//	simpleValueFunctionVis((ValueFunction)vi, p, initialState, domain, hashingFactory);
		//}

		return p;
	}



	private static Policy runQLearning(MountainCar mcGen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf,
			Environment env, boolean showPolicyMap) {
		final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

		System.out.println("//Q Learning Analysis//");

		QLearning agent = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			long startTime = System.nanoTime();

			agent = new QLearning(
				domain,
				0.999,
				hashingFactory,
				0.999, 0.999);
			
			for (int i = 0; i < numIterations; i++) {
				ea = agent.runLearningEpisode(env);
				env.resetEnvironment();
			}
			agent.initializeForPlanning(rf, tf, 1);
			p = agent.planFromState(initialState);
			AnalysisAggregator.addQLearningReward(calcRewardInEpisode(ea));
			AnalysisAggregator.addMillisecondsToFinishQLearning((int) (System.nanoTime()-startTime)/1000000);
			AnalysisAggregator.addStepsToFinishQLearning(ea.numTimeSteps());

		}
		AnalysisAggregator.printQLearningResults();
		//MapPrinter.printPolicyMap(getAllStates(domain,rf,tf,initialState), p, gen.getMap());
		System.out.println("\n\n");

		//visualize the value function and policy.
		/*if(showPolicyMap){
			simpleValueFunctionVis((ValueFunction)agent, p, initialState, domain, hashingFactory);
		}*/
		
		return p;

	}

	public static double calcRewardInEpisode(EpisodeAnalysis ea) {
		double myRewards = 0;

		//sum all rewards
		for (int i = 0; i<ea.rewardSequence.size(); i++) {
			myRewards += ea.rewardSequence.get(i);
		}
		return myRewards;
	}



	public static void main(String[] args) {

		MountainCar mcGen = new MountainCar();
		Domain domain = mcGen.generateDomain();
		TerminalFunction tf = new MountainCar.ClassicMCTF();
		RewardFunction rf = new GoalBasedRF(tf, 100);
		Environment env;
		
		
		//HashableStateFactory hashingFactory;
		//hashingFactory = new SimpleHashableStateFactory();

		StateGenerator rStateGen = new MCRandomStateGenerator(domain);
		SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
		State initialState = mcGen.getCleanState(domain);
		SARSData dataset = collector.collectNInstances(rStateGen, rf, 5000, 20, tf, null);
		
		//env = new SimulatedEnvironment(domain, rf, tf, rStateGen);
		env = new SimulatedEnvironment(domain, rf, tf,
				MountainCar.getCleanState(domain, mcGen.physParams));

		//Policy p = runQLearning(mcGen, domain, initialState, rf, tf, env, false);
		
		Policy p = runValueIteration(mcGen, domain, initialState, rf, tf, false);

		
		//
		// QLearning
		//
		/*
		//qLearningExample(domain, hashingFactory, rf, tf, rStateGen);
		QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0., 1.);

		for(int i = 0; i < 1; i++){
			EpisodeAnalysis ea = agent.runLearningEpisode(env);

			//ea.writeToFile(outputPath + "ql_" + i);
			System.out.println("qLearningExample: " + i + ": " + ea.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}
		agent.initializeForPlanning(rf, tf, 1);
		Policy p = agent.planFromState(initialState);*/
		
		

		/*ConcatenatedObjectFeatureVectorGenerator fvGen = new ConcatenatedObjectFeatureVectorGenerator(true,
				MountainCar.CLASSAGENT);
		FourierBasis fb = new FourierBasis(fvGen, 4);

		LSPI lspi = new LSPI(domain, 0.99, fb, dataset);
		Policy p = lspi.runPolicyIteration(30, 1e-6);*/

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vob = new VisualActionObserver(domain, v);
		vob.initGUI();

		//SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf,
		//		MountainCar.getCleanState(domain, mcGen.physParams));
		EnvironmentServer envServ = new EnvironmentServer(env, vob);

		for(int i = 0; i < 1; i++){
			p.evaluateBehavior(envServ);
			envServ.resetEnvironment();
		}

		System.out.println("Finished");

	}

}
