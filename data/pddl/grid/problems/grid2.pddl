(define (problem small)
  (:domain grid)

  (:objects node0-0 node0-1
  		node1-0 node1-1
	    triangle diamond square circle
	    key0)

  (:init (place node0-0) (place node0-1)
	 (place node1-0) (place node1-1)
	 (shape triangle) (shape diamond) (shape square) (shape circle)
	 (key key0)
     (holding key0)
	 (conn node0-0 node0-1) (conn node0-1 node0-0)
	 (conn node1-0 node1-1) (conn node1-1 node1-0)
	 (conn node0-0 node1-0) (conn node1-0 node0-0)
	 (conn node0-1 node1-1) (conn node1-1 node0-1)
	 (open node0-0) (open node0-1)
	 (open node1-0) (open node1-1)
	 (key-shape key0 triangle)
	 (at-robot node0-0))
  (:goal (at key0 node0-0))
  )
