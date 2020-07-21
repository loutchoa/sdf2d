class minHeap {
	public:
		// Constructor
		minHeap(int N);
		// Destructor
		~minHeap();

		// Keys contains the distance values. The heap is organized 
		// according to these, using heap-sort mechanisms.
		double* Keys;
		// H2P contains the index of the point for each element in the heap
		int* H2P;
		// P2H contains the position in the heap for each point (-1) if not in the heap
		int* P2H;

		/**
		 * return true if vertex i in the heap
		 * @param i vertex we want to check
		 */
		bool isInHeap(int i);

		/**
		 * return the number of elements in the heap
		*/
		int size();

		/**
		 * add a new element in the queue, keep priority queue structure
		 * @param i vertex which added to the queue
		 * @param d his distance
		 */
		void insert(int i, double d);

		/**
		 * update distance value, keep priority queue structure
		 * @param i vertex which distance will be decreased
		 * @param d new distance
		 */
		void decrease(int i, double d);

		/**
		 * remove the top (smallest) element from the queue, return the index
		 */
		int  extractMin();

		/**
		 * Print values of the heap
		 */
		void print();

	private:

		// Max size of the heap
		int max_size;

		// Keeps track of the number of elements in heap
		int heapSize;

		/**
		 * heapify from a given position
		 * @param i: root of subtree which must be heapified
		 */
		void heapify(int Ind);

		/**
		 * swap vertices pointers and update the idx fields
		 * @param i: first vertex
		 * @param j: second vertex
		 */
		void swap(int i, int j);

		/**
		 * Parent node to node i, convenience,
		 * should be inlined...
		 * i should be > 0, otherwise a bug?
		 */
		static inline int parent(int i) {
			return (i > 0) ? (i - 1) / 2 : 0;
		}

		/**
		 * Left node to node i, convenience,
		 * should be inlined...
		 */
		static inline int left(int i) {
			return 2 * i + 1;
		}

		/**
		 * Right node to node i, convenience,
		 * should be inlined...
		 */
		static inline int right(int i) {
			return 2 * i + 2;
		}
};