classdef MinHeap < Heap
%--------------------------------------------------------------------------
% Class:        MinHeap < Heap (& handle)
%               
% Constructor:  H = MinHeap(n);
%               
% Properties:   (none)
%               
% Methods:                 H.InsertKey(keyvalue);
%                          H.Update(keyvalue);
%               sx       = H.Sort();
%               min      = H.ReturnMin();
%               min      = H.ExtractMin();
%               count    = H.Count();
%               capacity = H.Capacity();
%               bool     = H.IsEmpty();
%               bool     = H.IsFull();
%                          H.Clear();
%               
% Description:  This class implements a min-heap of numeric keys-values
%               
% Author:       Brian Moore
%               brimoor@umich.edu
%               
% Modified by : Fraissinet-Tachet Yohan,
%--------------------------------------------------------------------------
    %
    % Public methods
    %
    methods (Access = public)
        %
        % Constructor
        %
        function this = MinHeap(varargin)
            %----------------------- Constructor --------------------------
            % Syntax:       H = MinHeap(n);
            %               
            % Inputs:       n is the maximum number of keys-values that H can hold
            %               
            % Description:  Creates a min-heap with capacity n
            %--------------------------------------------------------------
            
            % Call base class constructor
            this = this@Heap(varargin{:});
            
            % Construct the min heap
            this.BuildMinHeap();
        end
        
        %
        % Insert key
        %
        function InsertKey(this,kv)
            %------------------------ InsertKey ---------------------------
            % Syntax:       H.InsertKey(kv);
            %               
            % Inputs:       kv is a struct with kv.key the key (a number)
            %                              and  kv.value the value
            %                              (whatever data struct, I use it
            %                              with coordinates [x y])
            %
            % Description:  Inserts key-value into H
            %--------------------------------------------------------------
            
            this.SetLength(this.k + 1);
            this.x(this.k).key = inf;
            this.DecreaseKey(this.k,kv);
        end
        
        %
        % Update key of a value
        %
        function Update(this,kv)
            %------------------------ InsertKey ---------------------------
            % Syntax:       H.Update(kv);
            %               
            % Inputs:       kv is a struct with kv.key the key (a number)
            %                              and  kv.value the value
            %                              (whatever data struct, I use it
            %                              with coordinates [x y])
            %
            % Description:  Inserts key-value into H
            %--------------------------------------------------------------
            % Looking for the index of the old kv in x but seems not
            % perfect since we need to construct the table and look into it
            Table = struct2table(this.x(1:this.k)); 
            [~, index]=ismember(kv.value,Table.value,'rows');
            this.DecreaseKey(index,kv);
        end
        
        %
        % Sort the heap
        %
        function sx = Sort(this)
            %-------------------------- Sort ------------------------------
            % Syntax:       sx = H.Sort();
            %               
            % Outputs:      sx is a vector taht contains the sorted
            %               (ascending order) keys in H
            %               
            % Description:  Returns the sorted values in H
            %--------------------------------------------------------------
            
            % Sort the heap
            nk = this.k; % virtual heap size during sorting procedure
            for i = this.k:-1:2
                this.Swap(1,i);
                nk = nk - 1;
                this.MinHeapify(1,nk);
            end
            this.x(1:this.k) = flipud(this.x(1:this.k));
            sx = this.x(1:this.k);
        end
        
        %
        % Return minimum element
        %
        function min = ReturnMin(this)
            %------------------------ ReturnMin ---------------------------
            % Syntax:       min = H.ReturnMin();
            %               
            % Outputs:      min is the minimum key in H
            %               
            % Description:  Returns the minimum key in H
            %--------------------------------------------------------------
            
            if (this.IsEmpty() == true)
                min = [];
            else
                min = this.x(1);
            end
        end
        
        %
        % Extract minimum element
        %
        function min = ExtractMin(this)
            %------------------------ ExtractMin --------------------------
            % Syntax:       min = H.ExtractMin();
            %               
            % Outputs:      min is the minimum key-value in H
            %               
            % Description:  Returns the minimum key-value in H and extracts it
            %               from the heap
            %--------------------------------------------------------------
            
            this.SetLength(this.k - 1);
            min = this.x(1);
            this.x(1) = this.x(this.k + 1);
            this.MinHeapify(1);
        end
    end
    
    %
    % Private methods
    %
    methods (Access = private)
        %
        % Decrease key at index i
        %
        function DecreaseKey(this,i,kv)
            if (i > this.k)
                % Index overflow error
                MinHeap.IndexOverflowError();
            elseif (kv.key > this.x(i).key)
                % Decrease key error
                MinHeap.DecreaseKeyError();
            end
            this.x(i) = kv;
            while ((i > 1) && (this.x(Heap.parent(i)).key > this.x(i).key))
                this.Swap(i,Heap.parent(i));
                i = Heap.parent(i);
            end
        end
        
        %
        % Build the min heap
        %
        function BuildMinHeap(this)
            for i = floor(this.k / 2):-1:1
                this.MinHeapify(i);
            end
        end
        
        %
        % Maintain the min heap property at a given node
        %
        function MinHeapify(this,i,size)
            % Parse inputs
            if (nargin < 3)
                size = this.k;
            end
            
            ll = Heap.left(i);
            rr = Heap.right(i);
            if ((ll <= size) && (this.x(ll).key < this.x(i).key))
                smallest = ll;
            else
                smallest = i;
            end
            if ((rr <= size) && (this.x(rr).key < this.x(smallest).key))
                smallest = rr;
            end
            if (smallest ~= i)
                this.Swap(i,smallest);
                this.MinHeapify(smallest,size);
            end
        end
    end
    
    %
    % Private static methods
    %
    methods (Access = private, Static = true)
        %
        % Decrease key error
        %
        function DecreaseKeyError()
            error('You can only decrease keys in MinHeap');
        end
        
        %
        % Index overflow error
        %
        function IndexOverflowError()
            error('MinHeap index overflow');
        end
    end
end