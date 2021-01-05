import Neuron from "./Neuron"

export let connections = 0

/**
 * 2个神经元之间的连接
 */
export default class Connection {
	static uid() {
		return connections++
	}
	/**
	 * 
	 * @param {Neuron} from 输入神经元
	 * @param {Neuron} to 输出神经元
	 * @param {number} weight 连接的权重
	 */
	constructor(from, to, weight) {
		if (!from || !to)
			throw new Error("Connection Error: Invalid neurons")

		/**
		 * 唯一id
		 */
		this.ID = Connection.uid()
		/**
		 * 输入神经元
		 */
		this.from = from
		/**
		 * 输出神经元
		 */
		this.to = to
		/**
		 * 权重
		 */
		this.weight = typeof weight == 'undefined' ? Math.random() * .2 - .1 : weight
		/**
		 * the separate, adaptive learning rate for each connection（针对网络中每个连接的自适应学习步长）
		 * 其思想是在神经网络的每一个连接处都应该有该连接自己的自适应学习步长
		 * 并在我们调整该连接对应的参数时调整自己的学习步长：如果权值参数修正梯度，那就应该减小步长；反之，应该增大步长。
		 * 下图给出了intuition。我的理解是在多层神经网络中，不同层的梯度通常相差悬殊，
		 * 最开始的几层对应的梯度可能比最后几层权值对应的梯度小几个数量级。
		 * 另外一方面，网络中每一单元又受其扇入单元的影响，为了修正一个同样的错误，各个单元的“学习步长”应该是不同的。
		 * 
		 * 一个可行的方法是有一个全局的学习步长
		 * 然后对每一个权值参数有一个local gain，用gij表示
		 * 初始时gij均取值为1，后每次迭代根据权值梯度的变化情况作出调整
		 */
		this.gain = 1
		this.gater = null
	}
}