classdef Heap < handle
%
% Abstract superclass for all heap classes
%
% Note: You cannot instantiate Heap objects directly; use MaxHeap or
%       MinHeap
%
    %
    % Protected properties
    %
    properties (Access = protected)
        k;                  % current number of elements
        n;                  % heap capacity
        x = struct([]);     % heap array of key-value
    end
    
    %
    % Public methods
    %
    methods (Access = public)
        %
        % Constructor
        %
        function this = Heap(n)
            % Initialize heap
            if (n == 0)
                Heap.ZeroCapacityError();
            end
            this.n = n;
            
            data.key = nan(1);
            data.value = nan(1,2);
            temp(1:n,1) = data;
            this.x = temp;
            
            % Empty heap
            this.Clear();
        end
        
        %
        % Return number of elements in heap
        %
        function count = Count(this)
            %-------------------------- Count -----------------------------
            % Syntax:       count = H.Count();
            %               
            % Outputs:      count is the number of values in H
            %               
            % Description:  Returns the number of values in H
            %--------------------------------------------------------------
            
            count = this.k;
        end
        
        %
        % Return heap capacity
        %
        function capacity = Capacity(this)
            %------------------------- Capacity ---------------------------
            % Syntax:       capacity = H.Capacity();
            %               
            % Outputs:      capacity is the size of H
            %               
            % Description:  Returns the maximum number of values that can 
            %               fit in H
            %--------------------------------------------------------------
            
            capacity = this.n;
        end
        
        %
        % Check for empty heap
        %
        function bool = IsEmpty(this)
            %------------------------- IsEmpty ----------------------------
            % Syntax:       bool = H.IsEmpty();
            %               
            % Outputs:      bool = {true,false}
            %               
            % Description:  Determines if H is empty
            %--------------------------------------------------------------
            
            if (this.k == 0)
                bool = true;
            else
                bool = false;
            end
        end
        
        %
        % Check for full heap
        %
        function bool = IsFull(this)
            %-------------------------- IsFull ----------------------------
            % Syntax:       bool = H.IsFull();
            %               
            % Outputs:      bool = {true,false}
            %               
            % Description:  Determines if H is full
            %--------------------------------------------------------------
            
            if (this.k == this.n)
                bool = true;
            else
                bool = false;
            end
        end
        
        %
        % Clear the heap
        %
        function Clear(this)
            %-------------------------- Clear -----------------------------
            % Syntax:       H.Clear();
            %               
            % Description:  Removes all values from H
            %--------------------------------------------------------------
            
            this.SetLength(0);
        end
    end
    
    %
    % Abstract methods
    %
    methods (Abstract)
        %
        % Sort elements
        %
        Sort(this);
        
        %
        % Insert keyvalue
        %
        InsertKey(this,keyvalue);
    end
    
    %
    % Protected methods
    %
    methods (Access = protected)
        %
        % Swap elements
        %
        function Swap(this,i,j)
            kv = this.x(i);
            this.x(i) = this.x(j);
            this.x(j) = kv;
        end
        
        %
        % Set length
        %
        function SetLength(this,k)
            if (k < 0)
                Heap.UnderflowError();
            elseif (k > this.n)
                Heap.OverflowError();
            end
            this.k = k;
        end
    end
    
    %
    % Protected static methods
    %
    methods (Access = protected, Static = true)
        %
        % Parent node
        %
        function p = parent(i)
            p = floor(i / 2);
        end
        
        %
        % Left child node
        %
        function l = left(i)
            l = 2 * i;
        end
        
        % Right child node
        function r = right(i)
            r = 2 * i + 1;
        end
        
        %
        % Overflow error
        %
        function OverflowError()
            error('Heap overflow');
        end
        
        %
        % Underflow error
        %
        function UnderflowError()
            error('Heap underflow');
        end
        
        %
        % No capacity error
        %
        function ZeroCapacityError()
            error('Heap with no capacity is not allowed');
        end
    end
end