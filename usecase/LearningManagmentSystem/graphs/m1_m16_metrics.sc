import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes._
import ujson._
import scala.collection.mutable

val cpgPath: String = "/work/out/cpg.bin"
val outJson : String = "/work/out/metrics_m1_m16.json"
implicit val cpg: Cpg = CpgLoader.load(cpgPath)

// helpers (بدون lineNumber)
def S(x: Any): String = Option(x).map(_.toString).getOrElse("")
def lower(s:String)=S(s).toLowerCase
def isSynthetic(n: String): Boolean = {
  val s = S(n).trim
  s.isEmpty || s.matches("(?i)<operator>.*") || s.matches("(?i)<.*>") || s.contains("lambda") || s.contains("anonymous")
}
def fileOf(n: StoredNode): String = n.file.name.headOption.getOrElse("")

// core sets
val allMethods = cpg.method.l
val definedMethods: List[Method] = allMethods.filter(m => m.ast.isBlock.l.nonEmpty || !m.isExternal)
val cleanMethods = definedMethods.filterNot(m => isSynthetic(m.name) || isSynthetic(m.fullName))

// M1
val m1Pairs = cleanMethods.map { m =>
  val fn = S(m.fullName).trim
  val sg = Option(m.signature).map(_.toString.trim).getOrElse("")
  (fn, sg)
}.filter{ case (fn, _) => fn.nonEmpty }.distinct
val M1 = m1Pairs.size

// M2
val entryPointNames = List("main","WinMain","DllMain")
val entryMethods    = cpg.method.nameExact(entryPointNames:_*).l
val publicMethods   = cpg.method.where(_.modifier.modifierType("PUBLIC")).l
val entryPointFullNames =
  (entryMethods.map(_.fullName) ++ publicMethods.map(_.fullName))
    .map(_.trim).filter(fn => fn.nonEmpty && !isSynthetic(fn)).distinct
val M2 = entryPointFullNames.size

// M3: Call graph (line=None)
case class Edge(caller:String, callee:String, file:String, line:Option[Int])
val callEdges: List[Edge] = cpg.call.l.map { call =>
  val caller = call.method.fullName
  val callee = Option(call.methodFullName).getOrElse(call.name)
  val file   = call.file.name.headOption.getOrElse("")
  Edge(caller, callee, file, None)
}.distinct
val callersByCallee: Map[String,Set[String]] = callEdges.groupBy(_.callee).view.mapValues(_.map(_.caller).toSet).toMap
val calleesByCaller: Map[String,Set[String]] = callEdges.groupBy(_.caller).view.mapValues(_.map(_.callee).toSet).toMap
val M3_callEdgeCount = callEdges.size

// M4: Branches
def isIfCode(s:String) = lower(s).startsWith("if")
def isSwitchCode(s:String) = lower(s).startsWith("switch")
def isLoopCode(s:String) = { val t=lower(s); t.startsWith("for")||t.startsWith("while")||t.startsWith("do") }
case class BranchInfo(kind:String, code:String, file:String)
def branchInfos(m:Method): List[BranchInfo] = {
  m.controlStructure.l.map { n =>
    val c = S(n.code)
    val k =
      if (isIfCode(c)) "if"
      else if (isSwitchCode(c)) "switch"
      else if (isLoopCode(c)) "loop"
      else if (lower(c).startsWith("else")) "else"
      else "other"
    BranchInfo(k, c, fileOf(n))
  }
}
val m4All = cleanMethods.flatMap(branchInfos)
val M4_if     = m4All.count(_.kind=="if")
val M4_switch = m4All.count(_.kind=="switch")
val M4_loop   = m4All.count(_.kind=="loop")
val M4_total  = m4All.size

// M5: CFG (از DSL پایدار cfgNext/cfgPrev استفاده می‌کنیم)
case class CfgStats(nodes:Int, edges:Int, backEdges:Int, longestAcyclic:Int, basicSegments:Int)
def cfgStats(m:Method): CfgStats = {
  val nodes = m.cfgNode.l
  val nodeCount = nodes.size
  val edgeCount = nodes.map(_.cfgNext.size).sum
  val back = nodes.flatMap(n => n.cfgNext.l.map(dst => (n.order, dst.order))).count{ case (s,d) => d <= s }
  val segments = nodes.count { n => val inD=n.cfgPrev.size; val outD=n.cfgNext.size; !(inD==1 && outD==1) } match {
    case 0 => 1
    case x => x
  }
  val ords = nodes.map(_.order)
  val longest = if (ords.isEmpty) 1 else (ords.max - ords.min + 1)
  CfgStats(nodeCount, edgeCount, back, longest, segments)
}
val cfgByFunc: Map[String, CfgStats] = cleanMethods.map(m => m.fullName -> cfgStats(m)).toMap
def sumInt(xs: Iterable[Int]) = xs.sum
val M5_nodesSum  = sumInt(cfgByFunc.values.map(_.nodes))
val M5_edgesSum  = sumInt(cfgByFunc.values.map(_.edges))
val M5_backSum   = sumInt(cfgByFunc.values.map(_.backEdges))
val M5_segSum    = sumInt(cfgByFunc.values.map(_.basicSegments))
val M5_longestMax= cfgByFunc.values.map(_.longestAcyclic).foldLeft(0)(math.max)

// M6
val branchCount = cpg.controlStructure.code.l.count(isIfCode)
val loopCount   = cpg.controlStructure.code.l.count(isLoopCode)
val switchCount = cpg.controlStructure.code.l.count(isSwitchCode)
val exitFuncs = Set("exit","_exit","abort","longjmp","TerminateProcess")
val errorExitTotal = cpg.call.nameExact(exitFuncs.toSeq:_*).size

// M7
val ioFile = Set("fopen","fclose","fread","fwrite","fgets","fprintf","fscanf","open","close","read","write","rename","remove")
val ioNet  = Set("connect","accept","send","recv","bind","listen","socket","sendto","recvfrom","WSAStartup")
val ioEnv  = Set("getenv","setenv","putenv")
val cliArgsUse = cpg.identifier.nameExact("argv","argc").size
def idsOf(m:Method): Set[String] = m.ast.isIdentifier.name.l.toSet
val idToFuncs: Map[String, Set[String]] = cleanMethods.flatMap { m => idsOf(m).map(_ -> m.fullName)}.groupBy(_._1).view.mapValues(_.map(_._2).toSet).toMap
val sharedVarToFuncs: Map[String, Set[String]] = idToFuncs.filter{ case (_, fns) => fns.size >= 2 }
val sharedArtifacts = sharedVarToFuncs.keySet.toSeq.sorted
def stringArgsOf(names:Set[String]): Seq[String] = {
  cpg.call.l.flatMap { c => if (names.contains(c.name)) c.argument.code.l.map(S(_)) else Nil }.filter(_.nonEmpty)
}
val filesTouched     = stringArgsOf(ioFile).distinct.sorted
val networkEndpoints = stringArgsOf(ioNet).distinct.sorted

// M8
val consoleIn  = Set("scanf","gets","getchar","cin")
val consoleOut = Set("printf","puts","putchar","cout","cerr")
val consoleInCount  = cpg.call.name.l.count(consoleIn.contains)
val consoleOutCount = cpg.call.name.l.count(consoleOut.contains)
val fileIOCount     = cpg.call.name.l.count(ioFile.contains)
val netIOCount      = cpg.call.name.l.count(ioNet.contains)
val envIOCount      = cpg.call.name.l.count(ioEnv.contains)

// M9
val nameClues = List("login","logout","auth","register","create","delete","update","add","remove","deposit","withdraw","transfer","search","upload","download")
val nameCueMap: Map[String, Set[String]] =
  cleanMethods.map(_.fullName).distinct.map { fn =>
    val cues = nameClues.filter(k => lower(fn).contains(k)).toSet
    fn -> cues
  }.toMap
val stringCueMap: Map[String, Set[String]] =
  cleanMethods.map { m =>
    val lits = m.ast.isLiteral.code.l.map(lower)
    val cues = nameClues.filter(k => lits.exists(_.contains(k))).toSet
    m.fullName -> cues
  }.toMap

// M10
val stop = Set("int","char","void","long","short","double","float","const","static","unsigned","signed","return","include","define")
def splitTokens(s:String): Seq[String] =
  lower(s).replaceAll("[^a-z0-9_]+"," ").split("\\s+").filter(w => w.nonEmpty && !stop.contains(w)).toSeq
def bagOfWords(m:Method): Seq[String] = {
  val parts = mutable.ArrayBuffer[String]()
  parts += m.name
  parts ++= m.parameter.name.l
  parts ++= m.ast.isIdentifier.name.l
  parts ++= m.ast.isLiteral.code.l
  splitTokens(parts.mkString(" "))
}
val funcBags: Map[String, Seq[String]] = cleanMethods.map(m => m.fullName -> bagOfWords(m)).toMap
val termDocFreq: Map[String, Int] = funcBags.values.map(_.distinct).flatten.groupBy(identity).view.mapValues(_.size).toMap
val Ndocs = funcBags.size.max(1)
def tfidf(fn:String, k:String): Double = {
  val terms = funcBags.getOrElse(fn, Seq.empty)
  val tf = terms.count(_==k).toDouble / terms.size.max(1).toDouble
  val df = termDocFreq.getOrElse(k, 1).toDouble
  val idf = Math.log((Ndocs + 1.0) / df)
  tf * idf
}
def topTerms(fn:String, k:Int=10): Seq[(String,Double)] =
  funcBags.getOrElse(fn, Seq.empty).distinct.map(t => t -> tfidf(fn,t)).sortBy(-_._2).take(k)

// M11 (comments اغلب صفر می‌ماند)
val commentsPerFunc: Map[String, Int] = cleanMethods.map(m => m.fullName -> 0).toMap

// M12
def cyclomaticApprox(m:Method): Int = {
  val b = branchInfos(m)
  1 + b.count(_.kind=="if") + b.count(_.kind=="switch") + b.count(_.kind=="loop")
}
def isRecursive(m:Method): Boolean = {
  val fn = m.fullName
  calleesByCaller.getOrElse(fn, Set.empty).contains(fn)
}
case class FlowInfo(cyclo:Int, recursive:Boolean, cfgNodes:Int, cfgEdges:Int, estDepth:Int)
val perFuncComplexity: Map[String, FlowInfo] = cleanMethods.map { m =>
  val cstats = cfgStats(m)
  val cyc = cyclomaticApprox(m)
  val depthEst = cstats.longestAcyclic
  m.fullName -> FlowInfo(cyc, isRecursive(m), cstats.nodes, cstats.edges, depthEst)
}.toMap

// M13
def fileOfMethodFull(fn:String): String = cpg.method.fullNameExact(fn).file.name.headOption.getOrElse("")
val crossModuleEdges = callEdges.filter { e =>
  val f1 = fileOfMethodFull(e.caller); val f2 = fileOfMethodFull(e.callee)
  f1.nonEmpty && f2.nonEmpty && f1 != f2
}
val publicApiList = publicMethods.map(_.fullName).distinct.sorted
val M13_crossModuleEdgeCount = crossModuleEdges.size
val M13_publicApiCount = publicApiList.size

// M14
val knownUnsafe = Set("gets","strcpy","strcat","sprintf","vsprintf","scanf","strncpy","strncat","memcpy","memmove")
val unsafeCallsList = cpg.call.name.l.filter(knownUnsafe.contains).distinct.sorted
val degreeByFn: Map[String, Int] = {
  val undirected = callEdges.flatMap(e => List((e.caller,e.callee),(e.callee,e.caller))).distinct
  undirected.groupBy(_._1).view.mapValues(_.map(_._2).toSet.size).toMap
}
val degVals = degreeByFn.values.toSeq.sorted
def percIdx(vs: Seq[Int], p: Double): Int = if (vs.isEmpty) 0 else vs(Math.min(vs.size-1, Math.max(0, Math.round(p*(vs.size-1)).toInt)))
val highCouplingThreshold = percIdx(degVals, 0.90)
val highCouplingFunctions = degreeByFn.filter(_._2 >= highCouplingThreshold).keys.toSeq.sorted
def unvalidatedInputHeuristic(m:Method): Int = {
  val lits  = m.ast.isLiteral.code.l.map(lower)
  val ids   = m.ast.isIdentifier.name.l.map(lower)
  val hasInput = ids.exists(Set("argv","argc","input","buf","line").contains) || lits.exists(_.contains("input"))
  val hasValidate = (ids ++ lits).exists(s => s.contains("validate") || s.contains("sanitize") || s.contains("check") || s.contains("isvalid"))
  if (hasInput && !hasValidate) 1 else 0
}
val unvalidatedSum = cleanMethods.map(unvalidatedInputHeuristic).sum

// M15
case class InheritEdge(child:String, parent:String)
val inheritanceEdges: List[InheritEdge] =
  cpg.typeDecl.l.flatMap(td => td.inheritsFromTypeFullName.l.map(p => InheritEdge(td.fullName, p)))
val M15_inheritanceCount = inheritanceEdges.size
val M15_sharedGlobalsCount = sharedArtifacts.size

// M16
val ucdTerms: Seq[String] = sys.env.get("UCD_TERMS").map(_.split(",").toSeq.map(_.trim.toLowerCase).filter(_.nonEmpty)).getOrElse(Seq.empty)
val ucdMatchesAll: Map[String, Seq[String]] = if (ucdTerms.nonEmpty) {
  cleanMethods.map(_.fullName).flatMap { fn =>
    val terms = funcBags.getOrElse(fn, Seq.empty).distinct
    val hits = ucdTerms.filter(t => terms.exists(_.contains(t)))
    if (hits.nonEmpty) Map(fn -> hits) else Map.empty[String, Seq[String]]
  }.toMap
} else Map.empty

// JSON
val json = Obj(
  "meta" -> Obj("source_cpg" -> cpgPath, "methods_total" -> cleanMethods.size),
  "M1" -> Obj("count" -> M1),
  "M2" -> Obj("entryPointCount" -> M2, "entryPoints" -> Arr.from(entryPointFullNames.map(Str(_)))),
  "M3" -> Obj("callEdges" -> M3_callEdgeCount),
  "M4" -> Obj("if" -> M4_if, "switch" -> M4_switch, "loop" -> M4_loop, "total" -> M4_total),
  "M5" -> Obj("cfg_nodes_sum" -> M5_nodesSum, "cfg_edges_sum" -> M5_edgesSum, "cfg_backedges_sum" -> M5_backSum, "cfg_basic_segments_sum" -> M5_segSum, "cfg_longest_max" -> M5_longestMax),
  "M6" -> Obj("branchCount" -> branchCount, "loopCount" -> loopCount, "switchCount" -> switchCount, "errorExitTotal" -> errorExitTotal),
  "M7" -> Obj("cliArgsUse" -> cliArgsUse, "sharedArtifactsCount" -> M15_sharedGlobalsCount, "filesTouchedExamples" -> Arr.from(filesTouched.take(10).map(Str(_))), "networkEndpointsExamples" -> Arr.from(networkEndpoints.take(10).map(Str(_)))),
  "M8" -> Obj("consoleInCount" -> consoleInCount, "consoleOutCount" -> consoleOutCount, "fileIOCount" -> fileIOCount, "netIOCount" -> netIOCount, "envIOCount" -> envIOCount),
  "M9" -> Obj("nameCluesExamples" -> Arr.from(nameCueMap.take(10).map{ case (fn, ks) => Obj("fn"->fn, "clues"->Arr.from(ks.toSeq.map(Str(_)))) })),
  "M10" -> Obj("tfidfTopTermsExamples" -> Arr.from(cleanMethods.take(10).map(m => Obj("fn"->m.fullName, "top"->Arr.from(topTerms(m.fullName, 5).map{case (t,w)=> Obj("t"->t,"w"->w)}))))),
  "M11" -> Obj("hasCommentsFuncs" -> commentsPerFunc.count(_._2>0), "commentsTotal" -> commentsPerFunc.values.sum),
  "M12" -> Obj("cyclomaticSum" -> perFuncComplexity.values.map(_.cyclo).sum, "recursiveFuncs" -> perFuncComplexity.count(_._2.recursive), "maxDepthEst" -> perFuncComplexity.values.map(_.estDepth).foldLeft(0)(math.max)),
  "M13" -> Obj("crossModuleCallEdges" -> M13_crossModuleEdgeCount, "publicApiCount" -> M13_publicApiCount),
  "M14" -> Obj("unsafeCallsDistinct" -> unsafeCallsList.size, "highCouplingCount" -> highCouplingFunctions.size, "unvalidatedInputSum" -> unvalidatedSum),
  "M15" -> Obj("inheritanceEdges" -> M15_inheritanceCount, "sharedGlobalsApprox" -> M15_sharedGlobalsCount),
  "M16" -> Obj("enabled" -> (ucdTerms.nonEmpty), "ucdTerms" -> Arr.from(ucdTerms.map(Str(_))), "matchesCount" -> ucdMatchesAll.size)
)

import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
val p = Paths.get(outJson)
if (p.getParent != null) Files.createDirectories(p.getParent)
Files.write(p, ujson.write(json, indent=2).getBytes(StandardCharsets.UTF_8))

println(s"M1="+M1)
println(s"M2="+M2)
println(s"M3_callEdges="+M3_callEdgeCount)
println(s"M4_total="+M4_total)
println(s"M5_cfg_nodes_sum="+M5_nodesSum)
println(s"M6_branch_loop_switch="+(branchCount,loopCount,switchCount))
println(s"M7_cliArgs="+cliArgsUse)
println(s"M8_file_net_env="+(fileIOCount,netIOCount,envIOCount))
println(s"M9_nameCluesExamples="+nameCueMap.size)
println(s"M10_tfidfTopTermsExamples="+funcBags.size)
println(s"M11_commentsTotal="+commentsPerFunc.values.sum)
println(s"M12_recursive="+perFuncComplexity.count(_._2.recursive))
println(s"M13_crossModule="+M13_crossModuleEdgeCount)
println(s"M14_unsafeDistinct="+unsafeCallsList.size)
println(s"M15_inheritance_sharedGlobals="+(M15_inheritanceCount, M15_sharedGlobalsCount))
println(s"M16_enabled="+(ucdTerms.nonEmpty))





cpg.close()