;; 2016 states
;; 3x4 grid with two keys
(define (problem extra-large)
  (:domain grid)

  (:objects node0-0 node0-1 node0-2
  		node1-0 node1-1 node1-2
		node2-0 node2-1 node2-2
		node3-0 node3-1 node3-2
	    triangle diamond square circle
	    key0 key1)

  (:init (place node0-0) (place node0-1) (place node0-2)
	 (place node1-0) (place node1-1) (place node1-2)
	 (place node2-0) (place node2-1) (place node2-2)
	 (place node3-0) (place node3-1) (place node3-2)
	 (shape triangle) (shape diamond) (shape square) (shape circle)
	 (key key0) (key key1)
     (holding key0) (at key1 node2-1)
	 (conn node0-0 node0-1) (conn node0-1 node0-0)
	 (conn node1-0 node1-1) (conn node1-1 node1-0)
	 (conn node0-0 node1-0) (conn node1-0 node0-0)
	 (conn node0-1 node1-1) (conn node1-1 node0-1)
     (conn node0-1 node0-2) (conn node0-2 node0-1)
     (conn node1-1 node1-2) (conn node1-2 node1-1)
     (conn node0-2 node1-2) (conn node1-2 node0-2)
	 (conn node1-0 node2-0) (conn node2-0 node1-0)
	 (conn node1-1 node2-1) (conn node2-1 node1-1)
	 (conn node1-2 node2-2) (conn node2-2 node1-2)
	 (conn node2-0 node2-1) (conn node2-1 node2-0)
	 (conn node2-1 node2-2) (conn node2-2 node2-1)
	 (conn node2-0 node3-0) (conn node3-0 node2-0)
	 (conn node2-1 node3-1) (conn node3-1 node2-1)
	 (conn node2-2 node3-2) (conn node3-2 node2-2)
	 (conn node3-0 node3-1) (conn node3-1 node3-0)
	 (conn node3-1 node3-2) (conn node3-2 node3-1)
	 (open node0-0) (open node0-1) (open node0-2)
	 (open node1-0) (open node1-1) (open node1-2)
	 (open node2-0) (open node2-1) (open node2-2)
	 (open node3-0) (open node3-1) (open node3-2)
	 (key-shape key0 triangle)
	 (key-shape key1 diamond)
	 (at-robot node0-0))
  (:goal (and (at key0 node2-2) (at key1 node3-0)))
  )
