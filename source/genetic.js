/***********************************************************************************
/* 遗传算法实现
/* Genetic Algorithm implementation
/***********************************************************************************/

class GeneticAlgorithm {
	/**
	 * 
	 * @param {number} max_units 
	 * @param {number} top_units 
	 */
	constructor(max_units, top_units) {
		/**
		 * 人口中的最大单位数
		 * max number of units in population
		 */
		this.max_units = max_units

		/**
		 * 用于进化种群的顶级单位（优胜者）数量
		 * number of top units (winners) used for evolving population
		 */
		this.top_units = top_units

		if (this.max_units < this.top_units)
			this.top_units = this.max_units

		/**
		 * 当前人口中所有单位的数组
		 * array of all units in current population
		 */
		this.Population = []

		/**
		 * 用于缩放标准化输入值的因子
		 * the factor used to scale normalized input values
		 */
		this.SCALE_FACTOR = 200
	}
	/**
	 * 重置遗传算法参数
	 * resets genetic algorithm parameters
	 */
	reset() {
		/**
		 * 当前迭代数（等于当前总体数）
		 * current iteration number (it is equal to the current population number)
		 */
		this.iteration = 1
		/**
		 * 初始突变率
		 * initial mutation rate
		 */
		this.mutateRate = 1
		/**
		 * 最佳单位的人口数
		 * the population number of the best unit
		 */
		this.best_population = 0
		/**
		 * 最佳单位的体能
		 * the fitness of the best unit
		 */
		this.best_fitness = 0
		/**
		 * 史上最好的单位的分数
		 * the score of the best unit ever
		 */
		this.best_score = 0
	}
	/**
	 * 创建新的人口
	 * creates a new population
	 */
	createPopulation() {
		// 清除所有现有人口
		// clear any existing population
		this.Population.splice(0, this.Population.length)

		for (var i = 0; i < this.max_units; i++) {
			// 通过生成随机突触神经网络创建一个新单元
			// create a new unit by generating a random Synaptic neural network
			// 输入层2个，隐藏层6个，输出层1个
			// with 2 neurons in the input layer, 6 neurons in the hidden layer and 1 neuron in the output layer
			var newUnit = new synaptic.Architect.Perceptron(2, 6, 1)

			// 为新装置设置附加参数
			// set additional parameters for the new unit
			newUnit.index = i
			newUnit.fitness = 0
			newUnit.score = 0
			newUnit.isWinner = false

			// 将新单位添加到人口中
			// add the new unit to the population 
			this.Population.push(newUnit)
		}
	}
	/**
	 * 从人口中激活一个单位的神经网络
	 * activates the neural network of an unit from the population
	 * 根据输入计算输出动作
	 * to calculate an output action according to the inputs
	 * @param {*} bird
	 * @param {*} target
	 */
	activateBrain(bird, target) {
		// 输入1：鸟和目标之间的水平距离
		// input 1: the horizontal distance between the bird and the target
		var targetDeltaX = this.normalize(target.x, 700) * this.SCALE_FACTOR

		// 输入2：鸟和目标之间的高度差
		// input 2: the height difference between the bird and the target
		var targetDeltaY = this.normalize(bird.y - target.y, 800) * this.SCALE_FACTOR

		// 创建一个包含所有输入的数组
		// create an array of all inputs
		var inputs = [targetDeltaX, targetDeltaY]

		// 通过激活这只鸟的突触神经网络计算输出
		// calculate outputs by activating synaptic neural network of this bird
		var outputs = this.Population[bird.index].activate(inputs)

		// 如果输出大于0.5，则执行襟翼
		// perform flap if output is greater than 0.5
		if (outputs[0] > 0.5)
			bird.flap()
	}
	/**
	 * 通过对单元进行选择、交叉和突变来进化种群
	 * evolves the population by performing selection, crossover and mutations on the units
	 */
	evolvePopulation() {
		// 选择当前人口中排名靠前的单位来获得一组优胜者
		// select the top units of the current population to get an array of winners
		// （它们将被复制到下一个群体中）
		// (they will be copied to the next population)
		var Winners = this.selection()

		if (this.mutateRate == 1 && Winners[0].fitness < 0) {
			// 如果初始种群中的最佳单位具有负适应度
			// If the best unit from the initial population has a negative fitness 
			// 那就意味着没有任何鸟能到达第一道屏障！
			// then it means there is no any bird which reached the first barrier!
			// 扮演上帝的角色，我们可以摧毁这些坏人口，然后再尝试另一个。
			// Playing as the God, we can destroy this bad population and try with another one.
			this.createPopulation()
		} else {
			// 否则将突变率设置为实际值
			// else set the mutatation rate to the real value
			this.mutateRate = 0.2
		}

		// 用交叉和变异的新单位填充下一个种群的其余部分
		// fill the rest of the next population with new units using crossover and mutation
		for (var i = this.top_units; i < this.max_units; i++) {
			var parentA, parentB, offspring

			if (i == this.top_units) {
				// 后代是由两个最佳优胜者杂交而成
				// offspring is made by a crossover of two best winners
				parentA = Winners[0].toJSON()
				parentB = Winners[1].toJSON()
				offspring = this.crossOver(parentA, parentB)

			} else if (i < this.max_units - 2) {
				// 后代是由两个随机优胜者杂交而成
				// offspring is made by a crossover of two random winners
				parentA = this.getRandomUnit(Winners).toJSON()
				parentB = this.getRandomUnit(Winners).toJSON()
				offspring = this.crossOver(parentA, parentB)

			} else {
				// 后代是随机赢家
				// offspring is a random winner
				offspring = this.getRandomUnit(Winners).toJSON()
			}

			// 使后代变异
			// mutate the offspring
			offspring = this.mutation(offspring)

			// 利用子代的神经网络创建一个新单元
			// create a new unit using the neural network from the offspring
			var newUnit = synaptic.Network.fromJSON(offspring)
			newUnit.index = this.Population[i].index
			newUnit.fitness = 0
			newUnit.score = 0
			newUnit.isWinner = false

			// 通过用新单位替换旧单位来更新人口
			// update population by changing the old unit with the new one
			this.Population[i] = newUnit
		}

		// 如果冠军有史以来最好的健身，那就把它的成就储存起来吧！
		// if the top winner has the best fitness in the history, store its achievement!
		if (Winners[0].fitness > this.best_fitness) {
			this.best_population = this.iteration
			this.best_fitness = Winners[0].fitness
			this.best_score = Winners[0].score
		}

		// 按索引按升序排列新人口的单位
		// sort the units of the new population	in ascending order by their index
		this.Population.sort(function (unitA, unitB) {
			return unitA.index - unitB.index
		})
	}
	/**
	 * 从当前填充中选择最佳单位
	 * selects the best units from the current population
	 */
	selection() {
		// 按适合度降序排列当前人口的单位
		// sort the units of the current population	in descending order by their fitness
		var sortedPopulation = this.Population.sort(
			function (unitA, unitB) {
				return unitB.fitness - unitA.fitness
			}
		)

		// 把最优秀的单位标记为优胜者！
		// mark the top units as the winners!
		for (var i = 0; i < this.top_units; i++)
			this.Population[i].isWinner = true

		// 返回当前填充中的顶级单位数组
		// return an array of the top units from the current population
		return sortedPopulation.slice(0, this.top_units)
	}
	/**
	 * 在两个父对象之间执行单点交叉
	 * performs a single point crossover between two parents
	 * @param {*} parentA
	 * @param {*} parentB
	 */
	crossOver(parentA, parentB) {
		console.log('parentA', parentA)
		console.log('parentB', parentB)
		// 找一个交叉点
		// get a cross over cutting point
		var cutPoint = this.random(0, parentA.neurons.length - 1)

		// 在父母双方之间交换“偏见”信息：
		// swap 'bias' information between both parents:
		// 1. 从一个父对象复制到交叉点的左侧
		// 1. left side to the crossover point is copied from one parent
		// 2. 从第二个父对象复制交叉点后的右侧
		// 2. right side after the crossover point is copied from the second parent
		for (var i = cutPoint; i < parentA.neurons.length; i++) {
			var biasFromParentA = parentA.neurons[i]['bias']
			parentA.neurons[i]['bias'] = parentB.neurons[i]['bias']
			parentB.neurons[i]['bias'] = biasFromParentA
		}

		return this.random(0, 1) == 1 ? parentA : parentB
	}
	/**
	 * 对后代进行随机突变
	 * performs random mutations on the offspring
	 */
	mutation(offspring) {
		// 变异后代神经元的一些“偏差”信息
		// mutate some 'bias' information of the offspring neurons
		for (var i = 0; i < offspring.neurons.length; i++) {
			offspring.neurons[i]['bias'] = this.mutate(offspring.neurons[i]['bias'])
		}

		// 变异后代连接的一些“权重”信息
		// mutate some 'weights' information of the offspring connections
		for (var i = 0; i < offspring.connections.length; i++) {
			offspring.connections[i]['weight'] = this.mutate(offspring.connections[i]['weight'])
		}

		return offspring
	}
	/**
	 * 使基因突变
	 * mutates a gene
	 * @param {*} gene
	 */
	mutate(gene) {
		if (Math.random() < this.mutateRate) {
			var mutateFactor = 1 + ((Math.random() - 0.5) * 3 + (Math.random() - 0.5))
			gene *= mutateFactor
		}

		return gene
	}
	random(min, max) {
		return Math.floor(Math.random() * (max - min + 1) + min)
	}
	getRandomUnit(array) {
		return array[this.random(0, array.length - 1)]
	}
	normalize(value, max) {
		// 将值限制在其最小/最大限制之间
		// clamp the value between its min/max limits
		if (value < -max)
			value = -max;
		else if (value > max)
			value = max

		// 规格化钳制值
		// normalize the clamped value
		return (value / max)
	}
}